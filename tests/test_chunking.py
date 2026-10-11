import pytest

from paperpilot.chunking import chunk_paper, clean_text, split_words


def test_split_words_overlap():
    words = [str(i) for i in range(10)]
    windows = split_words(words, size=4, overlap=1)
    assert windows[0] == ["0", "1", "2", "3"]
    assert windows[1][0] == "3"            # overlap carries the last word over
    assert windows[-1][-1] == "9"          # nothing is lost at the end


def test_split_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_words(["a"], size=3, overlap=3)


def test_clean_text_joins_hyphenation():
    assert clean_text("reser-\nvoir   computing\n") == "reservoir computing"


def test_chunks_keep_page_numbers_and_stop_at_references():
    long_text = " ".join(["word"] * 60)
    pages = [(1, long_text), (2, long_text + "\nReferences\n[1] Some cited paper"), (3, long_text)]
    chunks = chunk_paper("p1", "Title", pages, size=30, overlap=5)
    assert {c.page for c in chunks} == {1, 2}       # page 3 comes after References
    assert all("cited paper" not in c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # ids are unique
