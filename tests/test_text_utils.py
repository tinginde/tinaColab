from my_rag_project.utils.text_utils import Document, read_split_md


def test_read_split_md_basic():
    md = "# Header1\nText1\n## Subheader\nText2"
    docs = read_split_md(md)
    assert len(docs) == 2
    assert docs[0].metadata["header"] == "Header1"
    assert docs[0].page_content == "Text1"
    assert docs[1].metadata["header"] == "Subheader"
    assert docs[1].page_content == "Text2"
