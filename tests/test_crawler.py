"""Crawler tests. They run against a tiny fake website served from localhost,
so no real site is contacted.

    python -m unittest discover -s tests -t . -v
"""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from ais_assistant import crawler as c

FOOTER = "<footer><p>AIS Global HQ, Example Street</p><p>Offices: Noida, India and Austin, USA</p></footer>"
NAV = '<nav><a href="/">Home</a><a href="/about">About</a><a href="/contact">Contact</a></nav>'
LOREM = "This sentence pads the page so it clears the minimum text length. " * 5


def page(title: str, body: str, extra_links: str = "") -> str:
    return (f"<html><head><title>{title}</title><script>var secret='JS_NOISE';</script></head>"
            f"<body>{NAV}<h1>{title}</h1><p>{body}</p><p>{LOREM}</p>{extra_links}{FOOTER}</body></html>")


class FakeSite(BaseHTTPRequestHandler):
    """Routes: / links to most pages; /hidden exists only in the sitemap."""

    requests_seen: list[tuple[str, float]] = []
    hosts_seen: list[tuple[str, str]] = []   # (Host header, path)

    def log_message(self, format, *args):  # keep test output clean
        pass

    def do_GET(self):
        import time
        FakeSite.requests_seen.append((self.path, time.monotonic()))
        FakeSite.hosts_seen.append((self.headers.get("Host", ""), self.path))
        port = self.server.server_port # type: ignore
        routes = {
            "/robots.txt": ("text/plain",
                            f"User-agent: *\nDisallow: /private\nSitemap: http://127.0.0.1:{port}/sitemap.xml\n"),
            "/sitemap.xml": ("application/xml",
                             f"<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                             f"<url><loc>http://127.0.0.1:{port}/hidden</loc></url></urlset>"),
            "/": ("text/html", page("Home", "Welcome to AIS home.", "".join(
                f'<a href="/{p}">{p}</a>' for p in
                ("about", "services", "careers", "contact", "private/secret", "logo.png", "dup", "about#team"))
                + '<a href="http://example.org/jobs">Jobs elsewhere</a>')),
            "/about": ("text/html", page("About", "AIS is a technology company serving many industries.")),
            "/services": ("text/html", page("Services", "AIS offers data and AI engineering services.")),
            "/careers": ("text/html", page("Careers", "Apply for jobs through the careers page.")),
            "/contact": ("text/html", page("Contact", "Reach AIS through the contact page.")),
            "/hidden": ("text/html", page("Hidden", "Only reachable through the sitemap.")),
            "/private/secret": ("text/html", page("Secret", "Robots.txt forbids this page.")),
            # Same visible text as /about under another URL: must be de-duplicated.
            "/dup": ("text/html", page("About", "AIS is a technology company serving many industries.")),
        }
        if self.path in routes:
            ctype, body = routes[self.path]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()


class CrawlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSite)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def run_crawl(self, delay=0.05, max_pages=50, extra_hosts=()):
        FakeSite.requests_seen = []
        FakeSite.hosts_seen = []
        crawler = c.Crawler(self.base, delay=delay, max_pages=max_pages, extra_hosts=extra_hosts)
        return crawler, crawler.crawl()

    # -- behaviour ----------------------------------------------------------- #
    def test_collects_linked_and_sitemap_pages(self):
        _, pages = self.run_crawl()
        paths = {p.url.replace(self.base.rstrip("/"), "") or "/" for p in pages}
        self.assertTrue({"/", "/about", "/services", "/careers", "/contact", "/hidden"} <= paths)

    def test_obeys_robots_txt(self):
        crawler, pages = self.run_crawl()
        fetched = [path for path, _ in FakeSite.requests_seen]
        self.assertNotIn("/private/secret", fetched)
        self.assertEqual(len(crawler.robots_blocked), 1)

    def test_skips_non_pages_and_duplicates(self):
        crawler, pages = self.run_crawl()
        fetched = [path for path, _ in FakeSite.requests_seen]
        self.assertNotIn("/logo.png", fetched)                        # file extension skipped
        self.assertEqual(fetched.count("/about"), 1)                  # '#team' variant merged
        self.assertTrue(any(s["reason"] == "duplicate content" for s in crawler.skipped))

    def test_respects_rate_limit(self):
        self.run_crawl(delay=0.25, max_pages=4)
        times = [t for _, t in FakeSite.requests_seen]
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertGreater(len(gaps), 3)
        self.assertGreaterEqual(min(gaps), 0.25 * 0.95)

    def test_respects_max_pages(self):
        _, pages = self.run_crawl(max_pages=3)
        self.assertEqual(len(pages), 3)

    def test_strips_scripts_and_menus(self):
        _, pages = self.run_crawl()
        everything = "\n".join(p.text for p in pages)
        self.assertNotIn("JS_NOISE", everything)
        self.assertNotIn("Home\nAbout\nContact", everything)

    def test_footer_kept_once_on_contact_page(self):
        _, pages = self.run_crawl()
        c.split_boilerplate(pages)
        everything = "\n".join(p.text for p in pages)
        self.assertEqual(everything.count("Offices: Noida, India and Austin, USA"), 1)
        contact = next(p for p in pages if p.url.endswith("/contact"))
        self.assertIn("Offices: Noida, India and Austin, USA", contact.text)

    def test_footer_goes_to_main_contact_page_not_a_sub_page(self):
        mk = lambda path, extra: c.Page(f"https://x.com{path}", "t", f"Footer line\n{extra}", "", "")
        pages = [mk("/contact-us/inquiry", "a"), mk("/about", "b"), mk("/contact-us", "c")] + \
                [mk(f"/p{i}", str(i)) for i in range(4)]
        c.split_boilerplate(pages)
        self.assertIn("Footer line", pages[2].text)       # /contact-us
        self.assertNotIn("Footer line", pages[0].text)    # /contact-us/inquiry

    def test_records_external_links_without_following_them(self):
        crawler, _ = self.run_crawl()
        self.assertIn("http://example.org/jobs", crawler.external_links)
        self.assertNotIn("/jobs", [path for path, _ in FakeSite.requests_seen])

    def test_extra_host_gets_its_own_robots_check_and_is_crawled(self):
        # 'localhost:PORT' and '127.0.0.1:PORT' are different hosts that reach the same fake server.
        alias = f"localhost:{self.server.server_port}"
        crawler, _ = self.run_crawl(max_pages=50, extra_hosts=(alias,))
        robots_hosts = {host for host, path in FakeSite.hosts_seen if path == "/robots.txt"}
        self.assertEqual(len(robots_hosts), 2)                                   # one robots.txt per host
        self.assertTrue(any(host == alias for host, _ in FakeSite.hosts_seen))   # the extra host was visited
        self.assertNotIn("/private/secret", [path for _, path in FakeSite.hosts_seen])

    def test_unlisted_subdomain_is_not_crawled(self):
        crawler, _ = self.run_crawl()  # no extra hosts
        self.assertTrue(all(host == f"127.0.0.1:{self.server.server_port}" for host, _ in FakeSite.hosts_seen))

    def test_snapshot_round_trip(self):
        crawler, pages = self.run_crawl()
        with tempfile.TemporaryDirectory() as tmp:
            c.save_snapshot(pages, crawler, 0, Path(tmp))
            loaded = c.load_pages(Path(tmp) / "pages.jsonl")
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertEqual([p["url"] for p in loaded], [p.url for p in pages])
        self.assertEqual(manifest["pages_saved"], len(pages))

    def test_refuses_to_crawl_when_robots_unreachable(self):
        dead = c.Crawler("http://127.0.0.1:9/", delay=0.01, extra_hosts=())  # nothing listens on port 9
        with mock.patch.object(c.config, "MAX_RETRIES", 1):
            self.assertEqual(dead.crawl(), [])


class UrlHelperTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(c.normalize_url("HTTPS://AISinfo.com/About/#team"), "https://aisinfo.com/About")
        self.assertEqual(c.normalize_url("https://aisinfo.com/?utm_source=x"), "https://aisinfo.com/")
        self.assertEqual(c.normalize_url("https://aisinfo.com/p?id=3&utm_medium=y"), "https://aisinfo.com/p?id=3")

    def test_www_is_same_host(self):
        self.assertEqual(c.host_of("https://www.aisinfo.com/x"), c.host_of("https://aisinfo.com/y"))


if __name__ == "__main__":
    unittest.main()