from sungrid_copilot.ingestion.chunker import MAX_CHUNK_WORDS, chunk_markdown
from sungrid_copilot.ingestion.pipeline import load_chunks

DOC = """# SunGrid Cooperative — Example Notice

This preamble explains the scope of the whole document in enough words that it should
become its own overview chunk rather than being dropped as a bare title line.

## First Topic
Body one.

## Second Topic
Body two.
"""


def test_one_chunk_per_section_plus_overview():
    chunks = chunk_markdown("99_example", DOC, ["billing_account"])
    assert [c.section for c in chunks] == ["Overview", "First Topic", "Second Topic"]
    assert chunks[1].chunk_id == "99_example::first-topic"
    assert chunks[1].doc_title == "Example Notice"


def test_header_is_embedded_but_not_in_raw_text():
    chunk = chunk_markdown("99_example", DOC, ["billing_account"])[1]
    assert chunk.embed_text.startswith("[Example Notice > First Topic]\n")
    assert chunk.text == "Body one."


def test_short_preamble_is_dropped():
    chunks = chunk_markdown("99_example", "# Title\n\n## Only\nText.", ["company_updates"])
    assert [c.section for c in chunks] == ["Only"]


def test_oversized_section_is_split_by_paragraph():
    para = " ".join(["word"] * 200)
    doc = f"# Title\n\n## Long\n{para}\n\n{para}\n\n{para}"
    chunks = chunk_markdown("99_example", doc, ["company_updates"])
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= MAX_CHUNK_WORDS for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_real_corpus_chunks():
    chunks = load_chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    doc06 = [c for c in chunks if c.doc_id == "06_ambiguous_rebate_billing_adjustments"]
    assert doc06[0].section == "Overview"
    assert set(doc06[0].categories) == {"incentive_rebate", "billing_account"}
