"""Messages the worker receives and the catalog ingests, as typed models.

The outgoing models mirror the catalog's ingestion forms
(EvilFlowersCatalog ``apps/extraction/forms.py``): a payload that validates
here is one the catalog accepts. Boxes are page fractions with a top-left
origin on the original file's 1-based page.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Dimension of the catalog's `halfvec` column (`apps.extraction.models.EMBEDDING_DIMENSIONS`):
#: multilingual-e5-large-instruct, as served by tei-embed.
EMBEDDING_DIMENSIONS = 1024
#: The catalog keeps this much of a failed job's error.
MAX_ERROR_LENGTH = 2000


class Box(BaseModel):
    """``{x0, y0, x1, y1}`` as page fractions, origin top-left."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


# -- received ---------------------------------------------------------------------------


class LayoutRegion(BaseModel):
    """A region the analyzer's layout pass found (`tables` job input)."""

    model_config = ConfigDict(extra="ignore")

    page: int = Field(ge=1)
    seq: int = Field(ge=0)
    bbox: Box
    #: The printed caption next to the region, when the layout pass found one.
    caption: str | None = None


# -- sent to the catalog ------------------------------------------------------------------


class Region(BaseModel):
    kind: Literal["figure", "table", "equation"]
    page: int = Field(ge=1)
    seq: int = Field(ge=0)
    bbox: Box
    image_path: str | None = Field(None, max_length=255)
    #: Kept for catalogs that predate `printed_caption`; the same text.
    caption: str | None = None
    printed_caption: str | None = None
    description: str | None = None
    category: str | None = Field(None, max_length=50)
    markdown: str | None = None
    latex: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class Sentence(BaseModel):
    seq: int = Field(ge=0)
    #: Character offsets into the paragraph's text.
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    page: int = Field(ge=1)
    #: One box per line the sentence covers.
    quads: list[Box] = Field(default_factory=list)


class Paragraph(BaseModel):
    #: The embedding is set after extraction; its dimension is checked then too.
    model_config = ConfigDict(validate_assignment=True)

    page: int = Field(ge=1)
    seq: int = Field(ge=0)
    #: The innermost outline heading in force on the page.
    section: str | None = None
    bbox: Box
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    embedding: list[float] | None = None

    @field_validator("embedding")
    @classmethod
    def _dimensions(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(value)}.")
        return value

    @property
    def embedding_text(self) -> str:
        """The paragraph with its section title prefixed, as monad-knowledge's chunker embeds it."""
        return f"{self.section}\n{self.text}" if self.section else self.text


class ParagraphLink(BaseModel):
    #: `seq` of the paragraphs.
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    kind: Literal["next", "similar"]
    score: float | None = None


class JobReport(BaseModel):
    status: Literal["done", "failed"]
    error: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("error")
    @classmethod
    def _truncate(cls, value: str | None) -> str | None:
        return value[:MAX_ERROR_LENGTH] if value else None


def dump(model: BaseModel) -> dict[str, Any]:
    """JSON-ready, without the absent optional fields (the catalog treats them as unset)."""
    return model.model_dump(mode="json", exclude_none=True)
