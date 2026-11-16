"""Table regions of an extraction run (IP-018).

The analyzer's layout pass finds the tables; here each is read with
pdfplumber inside its box, turned into markdown, cropped to WebP and kept
with its position, instead of being dropped as the indexing pipeline does.

Each region also carries what is known about the quality of its markdown
(issue #19), in its ``provenance``: the grid signals of the read, the
review's outcome and the share of the markdown's values found in the PDF text
inside the box (``grounded_share``; values not found are likely invented).

With the judges on (issue #20), decider-vision first answers "is this a data table?" from the crop
(``decider.IS_TABLE``, through the kit's judges). An answer the kit bands ``hide`` (``JUDGE_THRESHOLDS``) skips the
Ollama review and hides the region; otherwise the region gets a **table score** (``table_score``) from the grounded
share, the grid, the review outcome and the decider's p, and a band from it once ``DECIDER_TABLE_BANDS`` is calibrated
(knowledge#65). The decision fields go on the region itself (the kit's ``Region``, evilflowers-worker#7;
EvilFlowersCatalog#189); with no answer the region is posted exactly as before (``contract.dump`` leaves out the
``None`` fields).
"""

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pdfplumber
import pymupdf

from evilflowers_text import metrics
from evilflowers_text.extraction import geometry
from evilflowers_text.extraction.decider import TableDecider, p_table
from evilflowers_text.extraction.table_review import Outcome, TableReviewer
from evilflowers_worker.contract import Box, LayoutRegion, Region
from evilflowers_worker.judges import Verdict

logger = logging.getLogger(__name__)

#: More rotated characters than this and the table is not read as text (landscape tables).
MAX_ROTATED_SHARE = 0.3
#: Below this share of filled cells the grid is mostly guesswork.
MIN_FILLED_SHARE = 0.35
#: Above this share of tiny cells ("Eve | nt", "ro") words were cut across columns.
MAX_FRAGMENT_SHARE = 0.4
#: pdfplumber's line-based default first, then the text-alignment strategy; the name goes into the region.
STRATEGIES: tuple[tuple[str, dict], ...] = (
    ("lines", {}),
    ("text", {"vertical_strategy": "text", "horizontal_strategy": "text"}),
)
#: A number as printed ("1,234.56", "0.05") or a run of letters; signs, stars and brackets are not compared.
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")

Rows = list[list[str | None]]
Band = Literal["keep", "review", "hide"]

#: Weights of the table score's inputs: the deterministic signals weigh most, the decider's p is one more input.
SCORE_WEIGHTS = {"grounded": 3.0, "grid": 2.0, "review": 1.0, "decider": 1.0}
#: What a review outcome says about the markdown. ``unusable`` and ``failed`` say nothing about it: left out.
REVIEW_SCORES: dict[str, float] = {"kept": 1.0, "rewritten": 0.5, "transcribed": 0.5}


@dataclass(frozen=True)
class Decision:
    """How the table score becomes a band (issue #20). Unset thresholds: no band from here.

    ``bands``: ``(hide_below, keep_from)`` on the table score; between them the band is ``review``. The "not a data
    table" gate is the kit's band of the ``is_table`` answer (``JUDGE_THRESHOLDS``), not this.
    """

    bands: tuple[float, float] | None = None

    def band(self, score: float | None) -> Band | None:
        if score is None or self.bands is None:
            return None
        hide_below, keep_from = self.bands
        if score >= keep_from:
            return "keep"
        return "hide" if score < hide_below else "review"


@dataclass(frozen=True)
class Grid:
    """pdfplumber's read of one table box: the markdown and the signals that decided it."""

    markdown: str | None
    #: The strategy whose grid was kept; None when none was plausible.
    strategy: str | None = None
    #: Of the kept grid, else of the last grid tried; None when no grid was found at all.
    filled_share: float | None = None
    fragment_share: float | None = None
    rotated_share: float = 0.0

    def signals(self) -> dict[str, Any]:
        return {
            "grid_strategy": self.strategy,
            "grid_filled_share": self.filled_share,
            "grid_fragment_share": self.fragment_share,
            "rotated_share": round(self.rotated_share, 3),
        }

    def decision_signals(self) -> dict[str, Any]:
        """The grid signals under the names of the catalog's ``decision_signals`` (EvilFlowersCatalog#189)."""
        return {
            "filled_share": self.filled_share,
            "fragment_share": self.fragment_share,
            "strategy": self.strategy,
            "rotated_share": round(self.rotated_share, 3),
        }

    def quality(self) -> float | None:
        """0–1 from the grid: filled and not fragmented. 0 for an implausible grid; None when no grid was found."""
        if self.filled_share is None:
            return None
        if self.strategy is None:
            return 0.0
        return self.filled_share * (1.0 - (self.fragment_share or 0.0))


def markdown(rows: Rows) -> str | None:
    """A GitHub-style markdown table; the first row is the header."""
    cells = [[" ".join((cell or "").split()).replace("|", "\\|") for cell in row] for row in rows if row]
    cells = [row for row in cells if any(row)]
    if not cells:
        return None
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


def rotated_share(chars: list[dict]) -> float:
    if not chars:
        return 0.0
    return sum(1 for char in chars if not char.get("upright", True)) / len(chars)


def shares(rows: Rows) -> tuple[float, float]:
    """The share of filled cells, and of filled cells that look like word fragments."""
    cells = [" ".join((cell or "").split()) for row in rows for cell in row]
    filled = [cell for cell in cells if cell]
    if not filled:
        return 0.0, 0.0
    fragments = [cell for cell in filled if len(cell) <= 3 and cell.isalpha() and cell.islower()]
    return len(filled) / len(cells), len(fragments) / len(filled)


def plausible(rows: Rows) -> bool:
    """False when the grid cut words apart or is mostly empty: better no markdown than garbage."""
    filled, fragments = shares(rows)
    return filled >= MIN_FILLED_SHARE and fragments <= MAX_FRAGMENT_SHARE


def table_area(plumber_page, box: Box):
    """The pdfplumber page cut to the table's box: only characters fully inside it."""
    width, height = float(plumber_page.width), float(plumber_page.height)
    bounds = (box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)
    return plumber_page.within_bbox(bounds, relative=False, strict=False)


def read_table(area) -> Grid:
    """Markdown of the table in `area`, None when it cannot be read reliably, with the signals behind it.

    The crop is kept either way, so a reader still sees the table as an image.
    """
    rotated = rotated_share(area.chars)
    if rotated > MAX_ROTATED_SHARE:
        return Grid(None, rotated_share=rotated)
    grid = Grid(None, rotated_share=rotated)
    for name, strategy in STRATEGIES:
        table = area.extract_table(strategy) if strategy else area.extract_table()
        if not table:
            continue
        filled, fragments = shares(table)
        found = plausible(table)
        grid = Grid(
            markdown(table) if found else None,
            name if found else None,
            round(filled, 3),
            round(fragments, 3),
            rotated,
        )
        if found:
            break
    return grid


def tokens(text: str) -> list[str]:
    """Numbers and words, compared the same way on both sides (NFKC folds ligatures and odd spaces)."""
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def grounded_share(table: str | None, text: str) -> float | None:
    """The share of the markdown's numbers and words found in `text`, the PDF text inside the table's box.

    A number must appear as a whole number (an invented 12 is not found in 2012). A word may also appear
    across a line break or hyphenation ("regres- sion"), so it is looked up in the text with the separators
    removed, so most words are found and the share moves mainly with the numbers. None when there is nothing to
    compare: no markdown, or no text layer (a scanned page). The caller also passes None for rotated text.
    """
    if not table:
        return None
    found = tokens(text)
    wanted = tokens(table)
    if not found or not wanted:
        return None
    numbers, joined = set(found), "".join(found)
    hits = sum(1 for token in wanted if token in numbers or (not token[0].isdigit() and token in joined))
    return round(hits / len(wanted), 3)


def table_score(grounded: float | None, grid: Grid, outcome: Outcome | None, p: float | None) -> float | None:
    """The weighted mean of the inputs that are known (``SCORE_WEIGHTS``), None when none is.

    Grounding weighs most (no invented values), the grid next; the review outcome and the decider's p are one input
    each. A missing input (no text layer, review off or failed, rotated text) is left out, not counted as 0.
    """
    inputs = {
        "grounded": grounded,
        "grid": grid.quality(),
        "review": REVIEW_SCORES.get(outcome) if outcome else None,
        "decider": p,
    }
    known = {name: value for name, value in inputs.items() if value is not None}
    if not known:
        return None
    weight = sum(SCORE_WEIGHTS[name] for name in known)
    return round(sum(SCORE_WEIGHTS[name] * value for name, value in known.items()) / weight, 3)


class TableReader:
    """Reads the analyzer's table regions of one PDF into catalog regions with a WebP crop each."""

    def __init__(
        self,
        storage_path: Path,
        *,
        dpi: int,
        quality: int,
        reviewer: TableReviewer | None = None,
        decider: TableDecider | None = None,
        decision: Decision | None = None,
    ):
        self.storage_path = storage_path
        self.dpi = dpi
        self.quality = quality
        self.reviewer = reviewer
        self.decider = decider
        self.decision = decision or Decision()

    def versions(self) -> dict[str, str]:
        """The libraries behind the markdown and the crops, for the run's provenance."""
        versions = {"extractor": "pdfplumber", "pdfplumber": pdfplumber.__version__, "pymupdf": pymupdf.VersionBind}
        if self.reviewer:
            versions["review_model"] = self.reviewer.model
        return versions

    def read(self, path: Path, output_dir: str, regions: Sequence[LayoutRegion]) -> list[Region]:
        """One region per readable table; a region that fails is logged and skipped, never the document."""
        kept = []
        with pdfplumber.open(path) as plumber, pymupdf.open(path) as document:
            for region in regions:
                try:
                    kept.append(self._region(plumber, document, output_dir, region))
                except Exception:
                    logger.exception("Table on page %s skipped", region.page)
        return kept

    def _region(self, plumber, document: pymupdf.Document, output_dir: str, region: LayoutRegion) -> Region:
        page = document[region.page - 1]
        image = geometry.render(page, geometry.to_rect(region.bbox, page), self.dpi)
        name = geometry.crop_name(output_dir, region.page, "tab", region.seq)
        geometry.save(image, self.storage_path / name, self.quality)
        area = table_area(plumber.pages[region.page - 1], region.bbox)
        grid = read_table(area)
        verdict = self.decider.is_table(image) if self.decider else None
        # The kit bands the answer only with calibrated thresholds for `is_table`: until then nothing is skipped.
        skipped = verdict is not None and verdict.band == "hide"
        text, outcome = grid.markdown, None
        if self.reviewer and not skipped:
            review = self.reviewer.review(image, text)
            text, outcome = review.markdown, review.outcome
            metrics.TABLE_REVIEWS.add(1, {"outcome": outcome})
        metrics.TABLES.add(1, {"markdown": "read" if text else "unreadable"})
        # Rotated text comes out of pdfplumber reversed ("tneiciffeoC"): a correct table would look invented.
        grounded = None if grid.rotated_share > MAX_ROTATED_SHARE else grounded_share(text, area.extract_text() or "")
        # The job binds `run_id` (and job, attempt) to every log line; Loki: `|= "Table read"`.
        logger.info(
            "Table read on page %s (seq %s): review %s, grounded share %s, grid %s",
            region.page,
            region.seq,
            "skipped" if skipped else outcome or "off",
            grounded,
            grid.strategy,
        )
        provenance = {**grid.signals(), "grounded_share": grounded}
        if outcome:
            provenance["review_outcome"] = outcome
        decided = self._decide(verdict, skipped, grid, grounded, outcome)
        return Region(
            kind="table",
            page=region.page,
            seq=region.seq,
            bbox=region.bbox,
            image_path=name,
            caption=region.caption,
            printed_caption=region.caption,
            markdown=text,
            provenance=provenance,
            **decided,
        )

    def _decide(
        self, verdict: Verdict | None, skipped: bool, grid: Grid, grounded: float | None, outcome: Outcome | None
    ) -> dict[str, Any]:
        """The decision fields of the region; none at all without a verdict (no answer): no band, as before.

        ``p`` and ``decision`` are the kit's: the probability of the answer given (``yes`` / ``no``), as the catalog
        stores them. The score takes the probability of "a data table" (``p_table``).

        A skipped region (not a data table) has no score: its PDF text grounds a table of contents perfectly, so a
        score would only let the catalog re-band it upward. The catalog re-bands it from ``p`` + ``decision`` instead.
        """
        if verdict is None:
            # Why there was none is counted by the kit (`text_judge_total{question="is_table", outcome}`).
            return {}
        score = None if skipped else table_score(grounded, grid, outcome, p_table(verdict))
        band: Band | None = "hide" if skipped else self.decision.band(score)
        metrics.TABLE_DECISIONS.add(1, {"band": band or "none"})
        return {
            **verdict.region_fields(),
            "band": band,
            "score": score,
            "review_outcome": outcome,
            "grounded_share": grounded,
            "signals": grid.decision_signals(),
        }
