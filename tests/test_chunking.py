"""Chunking tests, including a no-text-lost check on the real saved website."""

import re
import unittest

from ais_assistant import config
from ais_assistant.chunking import build_chunks, clean_title, split_text
from ais_assistant.crawler import load_pages


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("Hello\nWorld"), ["Hello\nWorld"])

    def test_respects_max_size(self):
        text = "\n".join(f"Line number {i} " + "x" * 60 for i in range(60))
        for chunk in split_text(text, max_chars=300):
            self.assertLessEqual(len(chunk), 300 + 120)  # + at most one repeated heading line

    def test_short_last_line_is_repeated_as_heading(self):
        body = "\n".join(["y" * 80] * 5)
        text = body + "\nWHO WE SERVE\n" + body
        chunks = split_text(text, max_chars=420, overlap_max=40)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("WHO WE SERVE"))

    def test_long_paragraph_is_split_at_sentences(self):
        text = " ".join(f"This is sentence number {i}." for i in range(40))
        chunks = split_text(text, max_chars=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_tiny_leftover_is_merged_back(self):
        text = "\n".join(["z" * 90] * 4 + ["tail"])
        chunks = split_text(text, max_chars=300, min_chars=50)
        self.assertEqual(len(chunks), 2)             # 3 lines + 1 line would be 2 ...
        self.assertTrue(chunks[-1].endswith("tail"))  # ... but "tail" must not be alone
        self.assertGreater(len(chunks[-1]), len("tail"))

    def test_empty_text(self):
        self.assertEqual(split_text(""), [])


class BuildChunksTests(unittest.TestCase):
    def page(self, url, title, body):
        return {"url": url, "title": title, "text": f"{title}\n{body}"}

    def test_title_line_removed_and_cleaned(self):
        chunks = build_chunks([self.page("https://a.com/x", "AIS Info | Banking", "b" * 300)])
        self.assertEqual(chunks[0].title, "Banking")
        self.assertNotIn("AIS Info", chunks[0].text)
        self.assertTrue(chunks[0].embed_text.startswith("Banking\n"))

    def test_identical_text_under_two_urls_is_indexed_once(self):
        body = "same words " * 40
        chunks = build_chunks([self.page("https://a.com/one", "T1", body),
                               self.page("https://a.com/two", "T2", body)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].also_at, ["https://a.com/two"])

    def test_near_empty_pages_are_dropped(self):
        self.assertEqual(build_chunks([self.page("https://a.com/form", "Form", "hi")]), [])

    def test_clean_title(self):
        self.assertEqual(clean_title("AIS Info | Who We Serve | Banking"), "Who We Serve | Banking")
        self.assertEqual(clean_title("AIS Info"), "AIS Info")


@unittest.skipUnless(config.PAGES_FILE.exists(), "run the crawler first")
class RealSnapshotTests(unittest.TestCase):
    def test_no_text_is_lost(self):
        """Every line of every kept page must appear in some chunk."""
        pages = load_pages()
        chunks = build_chunks(pages)
        norm = lambda s: re.sub(r"\s+", " ", s).lower().strip()  # same idea as the de-duplication
        indexed = norm(" ".join(c.text for c in chunks))
        # Lines of a de-duplicated page live in the chunk it duplicates.
        missing = []
        for p in pages:
            lines = p["text"].split("\n")[1:]
            if sum(len(l) for l in lines) < config.MIN_PAGE_CHARS // 2:
                continue
            missing += [(p["url"], l) for l in lines
                        if l.strip() and len(l) <= config.CHUNK_MAX_CHARS  # long lines are re-split on purpose
                        and norm(l) not in indexed]
        self.assertEqual(missing, [])

    def test_chunk_sizes_are_sane(self):
        chunks = build_chunks(load_pages())
        self.assertGreater(len(chunks), 50)
        # max_chars, plus at most one merged tiny leftover (< CHUNK_MIN_CHARS)
        limit = config.CHUNK_MAX_CHARS + config.CHUNK_MIN_CHARS
        self.assertTrue(all(len(c.text) <= limit for c in chunks))


if __name__ == "__main__":
    unittest.main()
