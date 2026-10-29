"""Crawler for the AIS website.

Downloads the public pages, pulls out the readable text and saves it under
data/raw/ so the rest of the project always works from the same copy.

    python -m ais_assistant.crawler
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from . import config

log = logging.getLogger("crawler")

# file types we never want to open as pages
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".rar", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".css", ".js", ".xml", ".json", ".woff", ".woff2", ".ttf",
)
TRACKING_PARAMS = {"fbclid", "gclid", "ref"}

# Tags that never hold real content. <nav> goes because the menu is the same on
# every page. I keep <header> and <footer>: the page title (h1) is sometimes inside
# <header> and the footer has the office addresses. Repeated text is dealt with
# later in split_boilerplate().
NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "nav", "template"]


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str
    sha256: str


# --- url helpers ---------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Make equivalent urls look identical, so we don't crawl the same page twice."""
    url, _ = urldefrag(url.strip())  # drop the #fragment
    parts = urlparse(url)

    # utm_* and friends are just tracking, the page is the same
    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ])

    path = parts.path.rstrip("/") or "/"  # "/about/" and "/about" are the same page
    return urlunparse((parts.scheme.lower(), parts.netloc.lower(), path, "", query, ""))


def host_of(url: str) -> str:
    """Host name without 'www.', so www.aisinfo.com and aisinfo.com count as one."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


# --- html -> text --------------------------------------------------------------

def parse_page(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Return (title, text, links) for one page."""
    soup = BeautifulSoup(html, "html.parser")

    # get the links first. The menu is how we find most pages, and it's removed below.
    links = [normalize_url(urljoin(base_url, str(a["href"]))) for a in soup.find_all("a", href=True)]

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # one paragraph / heading per line, no blank lines, no immediate repeats
    lines = []
    for raw in soup.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return title, "\n".join(lines), links


def split_boilerplate(pages: list[Page], min_pages: int = 6, threshold: float = 0.5) -> int:
    """Take out lines that appear on more than half of the pages (footer, menus, banners).

    If I leave them in, the same footer gets indexed once per page and fills the
    search results. They are put back once, on the main contact page, so things
    like the office addresses can still be found. Returns how many lines were moved.
    """
    # with only a few pages, "appears everywhere" doesn't tell us much
    if len(pages) < min_pages:
        return 0

    # on how many pages does each line appear?
    line_counts = Counter(line for p in pages for line in set(p.text.split("\n")))
    repeated = {line for line, n in line_counts.items() if n / len(pages) > threshold}
    if not repeated:
        return 0

    # keep one copy of each repeated line, in the order we first see them
    footer_lines = []
    for p in pages:
        lines = p.text.split("\n")
        for line in lines:
            if line in repeated and line not in footer_lines:
                footer_lines.append(line)
        p.text = "\n".join(line for line in lines if line not in repeated)

    # Put them on the main contact page. Shortest path = /contact-us, not a sub page
    # like /contact-us/deceased-data-inquiry (that gave the wrong source link before).
    contact_pages = [p for p in pages if "contact" in urlparse(p.url).path.lower()]
    target = min(contact_pages, key=lambda p: len(urlparse(p.url).path), default=pages[0])
    target.text = (target.text + "\n" + "\n".join(footer_lines)).strip()

    for p in pages:  # text changed, so the hashes have to be redone
        p.sha256 = hashlib.sha256(p.text.encode()).hexdigest()
    return len(repeated)


# --- crawler -------------------------------------------------------------------

class Crawler:
    def __init__(
        self,
        start_url: str = config.SITE_URL,
        delay: float = config.MIN_CRAWL_DELAY,
        max_pages: int = config.MAX_PAGES,
        user_agent: str = config.USER_AGENT,
        session: requests.Session | None = None,
        extra_hosts: tuple[str, ...] = config.EXTRA_HOSTS,
    ):
        self.start_url = normalize_url(start_url)
        # main site + subdomains we decided to include (careers.aisinfo.com)
        self.allowed_hosts = {host_of(self.start_url), *extra_hosts}
        self._robots = {}  # origin -> parsed robots.txt, every host has its own
        self._robots_unreachable: set[str] = set()
        self.delay = delay
        self.max_pages = max_pages
        self.user_agent = user_agent
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent

        self._last_request = 0.0
        # things that went wrong or were skipped, saved in manifest.json at the end
        self.robots_blocked: list[str] = []
        self.failed: list[dict] = []
        self.skipped: list[dict] = []
        self.external_links: set[str] = set()  # other sites. Only noted, never visited

    # -- requests ---------------------------------------------------------------

    def _wait_turn(self) -> None:
        """Sleep until at least `delay` seconds passed since the last request."""
        wait = self._last_request + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> requests.Response | None:
        """GET with the rate limit and a few retries. Returns None if it never worked."""
        for attempt in range(1, config.MAX_RETRIES + 1):
            self._wait_turn()
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                log.warning("%s: %s (attempt %d)", url, exc, attempt)
            else:
                # 5xx and 429 are usually temporary so we retry them.
                # Anything else (200, 404, ...) is a real answer.
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                log.warning("%s: HTTP %d (attempt %d)", url, resp.status_code, attempt)
            if attempt < config.MAX_RETRIES:
                time.sleep(2 ** attempt)  # wait longer after each failure
        return None

    # -- robots.txt and sitemap -------------------------------------------------

    def _robots_for(self, url: str) -> robotparser.RobotFileParser:
        """Rules for the host of this url. Downloaded once, then reused."""
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        return self._robots[origin]

    def _load_robots(self, origin: str) -> robotparser.RobotFileParser:
        rp = robotparser.RobotFileParser()
        resp = self._get(f"{origin}/robots.txt")

        if resp is not None and resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            crawl_delay = rp.crawl_delay(self.user_agent)
            if crawl_delay and float(crawl_delay) > self.delay:
                log.info("robots.txt asks for a Crawl-delay of %s, slowing down", crawl_delay)
                self.delay = float(crawl_delay)
        elif resp is not None and resp.status_code < 500:
            # 404 etc: the site has no robots.txt, so nothing is forbidden
            log.info("No robots.txt at %s (HTTP %d)", origin, resp.status_code)
            rp.parse([])
        else:
            # Couldn't read it at all. Safer to assume we're not allowed in.
            log.error("%s/robots.txt unreachable, not crawling this host", origin)
            self._robots_unreachable.add(origin)
        return rp

    def _sitemap_urls(self, home: str, rp: robotparser.RobotFileParser) -> list[str]:
        """Page urls listed in the sitemap. Follows sitemap index files too."""
        to_read = deque(rp.site_maps() or [urljoin(home, "/sitemap.xml")])
        already_read = set()
        found = []

        while to_read and len(already_read) < 20:  # cap so a weird site can't loop us forever
            sitemap_url = to_read.popleft()
            if sitemap_url in already_read:
                continue
            already_read.add(sitemap_url)

            resp = self._get(sitemap_url)
            if resp is None or resp.status_code != 200:
                continue
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                log.warning("Sitemap is not valid XML: %s", sitemap_url)
                continue

            # an index file points to other sitemaps, a normal one points to pages
            is_index = root.tag.endswith("sitemapindex")
            for el in root.iter():
                if el.tag.split("}")[-1] == "loc" and el.text:
                    (to_read if is_index else found).append(el.text.strip())
        return found

    # -- main loop --------------------------------------------------------------

    def _in_scope(self, url: str) -> bool:
        """Is this a page on one of our hosts, and not an image / pdf / etc?"""
        parts = urlparse(url)
        return (
            parts.scheme in ("http", "https")
            and host_of(url) in self.allowed_hosts
            and not parts.path.lower().endswith(SKIP_EXTENSIONS)
        )

    def crawl(self) -> list[Page]:
        # if the main site's robots.txt can't be read we don't crawl at all
        parts = urlparse(self.start_url)
        if f"{parts.scheme}://{parts.netloc}" in self._robots_unreachable:
            return []

        # every host gets its home page + whatever its sitemap lists
        scheme = urlparse(self.start_url).scheme
        start_pages = [self.start_url]
        for host in sorted(self.allowed_hosts):
            if host != host_of(self.start_url):
                start_pages.append(f"{scheme}://{host}/")

        queue = deque()  # urls still to visit
        queued = set()   # everything we ever added, so nothing is added twice
        for start in start_pages:
            rp = self._robots_for(start)
            parts = urlparse(start)
            if f"{parts.scheme}://{parts.netloc}" in self._robots_unreachable:
                continue
            candidates = [normalize_url(start)]
            candidates += [normalize_url(u) for u in self._sitemap_urls(start, rp)]
            for url in candidates:
                if self._in_scope(url) and url not in queued:
                    queued.add(url)
                    queue.append(url)
        log.info("Starting with %d urls (home pages + sitemaps)", len(queue))

        pages = []
        fetched = set()      # final urls after redirects
        seen_hashes = set()  # to catch the same text under two urls

        while queue and len(pages) < self.max_pages:
            url = queue.popleft()

            if not self._robots_for(url).can_fetch(self.user_agent, url):
                log.info("robots.txt disallows %s", url)
                self.robots_blocked.append(url)
                continue

            resp = self._get(url)
            if resp is None or resp.status_code != 200:
                reason = "no response" if resp is None else f"HTTP {resp.status_code}"
                self.failed.append({"url": url, "reason": reason})
                continue

            final_url = normalize_url(resp.url)  # may differ from url after a redirect
            if final_url in fetched or not self._in_scope(final_url):
                continue
            fetched.add(final_url)

            if "text/html" not in resp.headers.get("Content-Type", ""):
                self.skipped.append({"url": final_url, "reason": "not HTML"})
                continue

            title, text, links = parse_page(resp.text, final_url)

            for link in links:
                if self._in_scope(link):
                    if link not in queued:
                        queued.add(link)
                        queue.append(link)
                else:
                    link_parts = urlparse(link)
                    if link_parts.scheme in ("http", "https") and host_of(link) not in self.allowed_hosts:
                        self.external_links.add(link)

            # almost no text usually means the page is built with JavaScript
            if len(text) < config.MIN_PAGE_CHARS:
                self.skipped.append({"url": final_url, "reason": f"only {len(text)} chars of text"})
                log.warning("Skipping %s, almost no text (JavaScript page?)", final_url)
                continue

            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_hashes:
                self.skipped.append({"url": final_url, "reason": "duplicate content"})
                continue
            seen_hashes.add(digest)

            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            pages.append(Page(final_url, title, text, fetched_at, digest))
            log.info("[%d] %s (%d chars)", len(pages), final_url, len(text))

        if queue:
            log.warning("Stopped at max_pages=%d, %d urls not visited", self.max_pages, len(queue))
        return pages


# --- saving --------------------------------------------------------------------

def _write_atomic(path: Path, content: str) -> None:
    """Write to a temp file first, so a crash can't leave a half written snapshot."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def save_snapshot(pages: list[Page], crawler: Crawler, boilerplate_lines: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # pages.jsonl: one json object per line, one line per page
    jsonl = "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in pages)
    _write_atomic(out_dir / "pages.jsonl", jsonl)

    # manifest.json: how and when this snapshot was made, plus what we skipped
    manifest = {
        "start_url": crawler.start_url,
        "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_agent": crawler.user_agent,
        "delay_seconds": crawler.delay,
        "pages_saved": len(pages),
        "boilerplate_lines_moved": boilerplate_lines,
        "urls": [p.url for p in pages],
        "blocked_by_robots_txt": crawler.robots_blocked,
        "failed": crawler.failed,
        "skipped": crawler.skipped,
        "external_links_not_followed": sorted(crawler.external_links)[:200],
    }
    _write_atomic(out_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def load_pages(path: Path = config.PAGES_FILE) -> list[dict]:
    """Read the saved snapshot back. The indexing step starts from this."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl the AIS website into data/raw/")
    parser.add_argument("--url", default=config.SITE_URL)
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES)
    parser.add_argument("--delay", type=float, default=config.MIN_CRAWL_DELAY,
                        help="seconds between requests (never below 1.0)")
    parser.add_argument("--out", type=Path, default=config.RAW_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # the brief says one request per second at most, so --delay can't go lower
    crawler = Crawler(args.url, delay=max(args.delay, config.MIN_CRAWL_DELAY), max_pages=args.max_pages)
    pages = crawler.crawl()
    if not pages:
        raise SystemExit("Nothing was crawled. Check the internet connection and the log above.")

    moved = split_boilerplate(pages)
    save_snapshot(pages, crawler, moved, args.out)
    log.info("Saved %d pages to %s (%d repeated lines moved)", len(pages), args.out, moved)


if __name__ == "__main__":
    main()