"""A multimodal model checks each table read by pdfplumber (IP-018).

The model sees the table crop and pdfplumber's markdown draft. It keeps a
correct draft, or rewrites it from the image: split or merged columns, a
header spread over several lines, cut words. With no draft it transcribes
the table itself. Any failure keeps the draft.
"""

import base64
import io
import json
import logging

import requests
from PIL import Image

logger = logging.getLogger(__name__)

PROMPT = (
    "You see one table cropped from a page of a publication.\n"
    "{draft}\n"
    "Rules for the markdown table:\n"
    "- One markdown row per printed row, one column per printed column; never split a word or a value "
    "across columns and never merge two printed columns.\n"
    "- The header is one row: join a header printed over several lines or levels into one cell per "
    'column (e.g. "(1) Vertical subsidiaries"). Keep the stub column header (e.g. "Regressor").\n'
    "- A section title spanning the table (e.g. a row in italics) becomes its own row with the title in "
    "the first cell and the other cells empty.\n"
    "- Copy every value exactly as printed, with stars, signs and parentheses; leave empty cells empty; "
    "keep footnotes out.\n"
    'Answer only JSON: {{"ok": true}} when the draft is correct, else {{"ok": false, "markdown": "..."}}.'
)

DRAFT = "A text extractor read it as this markdown draft:\n\n{markdown}\n"
NO_DRAFT = "A text extractor could not read it. Transcribe it from the image."


def parse(content: str, draft: str | None) -> str | None:
    """The markdown to keep from the model's answer; the draft when the answer is unusable."""
    try:
        answer = json.loads(content[content.find("{") : content.rfind("}") + 1])
    except (ValueError, TypeError):
        return draft
    if not isinstance(answer, dict):
        return draft
    if answer.get("ok") is True and draft:
        return draft
    markdown = str(answer.get("markdown") or "").strip()
    if markdown.count("|") < 4 or "\n" not in markdown:
        return draft
    return markdown


class TableReviewer:
    """Asks an Ollama chat endpoint to check or rewrite one table."""

    def __init__(self, endpoint: str, model: str, *, api_key: str | None, timeout: float, max_side: int):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_side = max_side

    def _encode(self, image: Image.Image) -> str:
        image = image.copy()
        image.thumbnail((self.max_side, self.max_side))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def review(self, image: Image.Image, draft: str | None) -> str | None:
        prompt = PROMPT.format(draft=DRAFT.format(markdown=draft) if draft else NO_DRAFT)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": prompt, "images": [self._encode(image)]}],
                },
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
        except (requests.RequestException, ValueError):
            logger.exception("Table review failed, keeping the draft")
            return draft
        return parse(content, draft)
