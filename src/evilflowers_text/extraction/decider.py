"""Is this a data table? A judge of the kit answers from the crop (issue #20).

The question goes to ``decider-vision`` (Strands Decider 2B ``--vision``, knowledge#64) through the kit's ``judges``
(evilflowers-worker#6): its ``DeciderClient``, circuit breaker, per-job budget and ``<service>_judge_total`` metrics.
This module only says what is asked (``IS_TABLE``) and how the crop is sent.

``IS_TABLE`` is a yes/no question rejected on ``no``. With ``JUDGE_THRESHOLDS='{"is_table": {"keep": …, "hide": …}}'``
(knowledge#65) the kit bands the answer, and a ``hide`` there means "not a data table": the Ollama review is skipped.
Without thresholds the answer is never banded and nothing is skipped.

No answer (judges off, decider-vision down or its breaker open, the job's budget spent, an answer of an unknown
shape) is ``None``: the region then goes to the catalog as before issue #20, with no band.
"""

import io

from PIL import Image

from evilflowers_worker.judges import Binary, Judge, Panel, Verdict

#: The server scales the longer side to this many pixels anyway; scaling here keeps the request small.
MAX_SIDE = 448
STATE = "A region a layout model marked as a table."
IS_TABLE = Binary(
    id="is_table",
    instructions="Is this a data table (rows and columns of values), not a table of contents, a form or page layout?",
    reject=frozenset({"no"}),
)


def encode(image: Image.Image) -> bytes:
    """The crop as WebP, its longer side at most ``MAX_SIDE``: the kit sends images already encoded."""
    image = image.copy()
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=90)
    return buffer.getvalue()


def p_table(verdict: Verdict) -> float:
    """The probability that the crop is a data table (the kit's ``p`` is that of the answer given)."""
    return verdict.p if verdict.verdict == "yes" else 1.0 - verdict.p


class TableDecider:
    """Asks one job's panel whether a crop is a data table."""

    def __init__(self, judge: Judge, panel: Panel):
        self.judge = judge
        self.panel = panel

    def is_table(self, image: Image.Image) -> Verdict | None:
        return self.panel.ask(self.judge, IS_TABLE, state=STATE, images=[encode(image)])
