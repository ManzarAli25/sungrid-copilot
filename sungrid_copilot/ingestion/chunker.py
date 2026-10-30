"""Structure-aware chunking: one chunk per `##` section, with a context header.

The knowledge-base docs are short and already organised into self-contained `##`
sections (22–122 words each), so section boundaries are the chunk boundaries. Fixed-size
windows would merge unrelated sections and cut rules away from their conditions.
"""

import re
from dataclasses import dataclass, field

TITLE_PREFIX = "SunGrid Cooperative — "  # present on every doc title; noise in a header
MIN_PREAMBLE_WORDS = 20  # shorter preambles are just the title line
MAX_CHUNK_WORDS = 300  # ~400 tokens; size guard for future, less tidy documents


@dataclass
class Chunk:
    chunk_id: str  # stable: "<doc_id>::<section-slug>" — makes re-ingestion an upsert
    doc_id: str
    doc_title: str
    section: str
    text: str  # raw section body, shown to the LLM
    categories: list[str] = field(default_factory=list)

    @property
    def header(self) -> str:
        return f"[{self.doc_title} > {self.section}]"

    @property
    def embed_text(self) -> str:
        # The header carries the chunk's origin into the vector: many sections never
        # mention their own topic (e.g. doc 06 "Interaction with Autopay" never says "rebate").
        return f"{self.header}\n{self.text}"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_oversized(body: str) -> list[str]:
    """Split a section by paragraph if it exceeds MAX_CHUNK_WORDS; otherwise keep it whole."""
    if len(body.split()) <= MAX_CHUNK_WORDS:
        return [body]
    parts, current = [], []
    for para in re.split(r"\n\s*\n", body):
        if current and len(" ".join(current + [para]).split()) > MAX_CHUNK_WORDS:
            parts.append("\n\n".join(current))
            current = []
        current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(doc_id: str, markdown: str, categories: list[str]) -> list[Chunk]:
    lines = markdown.strip().splitlines()
    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("# ") else doc_id
    title = title.removeprefix(TITLE_PREFIX)
    body = "\n".join(lines[1:]) if title != doc_id else markdown

    preamble, *sections = re.split(r"^## ", body, flags=re.MULTILINE)
    raw_sections = []
    if len(preamble.split()) >= MIN_PREAMBLE_WORDS:
        raw_sections.append(("Overview", preamble.strip()))
    for section in sections:
        heading, _, text = section.partition("\n")
        raw_sections.append((heading.strip(), text.strip()))

    chunks, seen_ids = [], set()
    for heading, text in raw_sections:
        parts = _split_oversized(text)
        for i, part in enumerate(parts):
            label = heading if len(parts) == 1 else f"{heading} (part {i + 1})"
            chunk_id = f"{doc_id}::{_slugify(label)}"
            while chunk_id in seen_ids:  # guard against duplicate headings in one doc
                chunk_id += "-dup"
            seen_ids.add(chunk_id)
            chunks.append(Chunk(chunk_id, doc_id, title, label, part, list(categories)))
    return chunks
