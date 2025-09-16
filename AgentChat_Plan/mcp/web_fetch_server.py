# web_tools_mcp_server.py
# MCP stdio server exposing: fetch_page, follow_links, fetch_dynamic_page

from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from mcp.server import FastMCP
from mcp.types import TextContent
from playwright.async_api import async_playwright

# ---------------------- logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("web_tools_mcp_server")

# ---------------------- MCP server ----------------------
mcp_server = FastMCP(
    name="web_tools_mcp_server",
    instructions=(
        "Web tools: static HTML fetch, link listing, and dynamic rendering via Playwright."
    ),
)

# ---------------------- helpers ----------------------
USER_AGENT = (
    "WebToolsMCP/1.0 (+https://example.local) Python-HTTPX"
)


def _clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    h1 = soup.select_one("h1")
    return _clean_text(h1.get_text(" ", strip=True)) if h1 else ""


def _extract_text(soup: BeautifulSoup) -> str:
    return _clean_text(soup.get_text("\n", strip=True))


def _extract_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    for a in soup.select("a[href]"):
        txt = _clean_text(a.get_text(" ", strip=True))[:200]
        href = (a.get("href") or "").strip()
        if not href:
            continue
        href_abs = urljoin(base_url, href)
        links.append({"text": txt, "href": href_abs})
    # de-dup
    seen = set()
    out: List[Dict[str, str]] = []
    for l in links:
        if l["href"] in seen:
            continue
        seen.add(l["href"])
        out.append(l)
    return out


async def _fetch_html(url: str, timeout_s: float = 60.0, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        resp = await client.get(url)
        return resp


# ---------------------- Tool 1: fetch_page (static HTML) ----------------------
@mcp_server.tool()
async def fetch_page(
    url: str,
    css_selector: Optional[str] = None,
    text_only: bool = True,
    max_chars: int = 20000,
    timeout_s: float = 60.0,
    keep_nav: bool = False,
    keep_footer: bool = False,
    keep_script: bool = False,
) -> List[TextContent]:
    """Fetch static HTML and optionally extract text/HTML and links. Returns JSON as text content."""
    resp = await _fetch_html(url, timeout_s=timeout_s)
    status = resp.status_code
    raw = resp.text
    soup = BeautifulSoup(raw, "lxml")

    if not keep_script:
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
    if not keep_nav:
        for t in soup.select("nav, aside"):
            t.decompose()
    if not keep_footer:
        for t in soup.select("footer"):
            t.decompose()

    node = soup
    if css_selector:
        nodes = soup.select(css_selector)
        if nodes:
            frag_html = "\n".join(n.decode() for n in nodes)
            node = BeautifulSoup(frag_html, "lxml")
        else:
            node = BeautifulSoup("", "lxml")

    if text_only:
        content = _extract_text(node)
    else:
        content = _clean_text(node.decode())

    title = _extract_title(soup)
    links = _extract_links(soup, str(resp.url))

    if isinstance(content, str) and len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]"

    payload = {
        "url": str(resp.url),
        "status": status,
        "title": title,
        "content": content,
        "links": links,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


# ---------------------- Tool 2: follow_links ----------------------
@mcp_server.tool()
async def follow_links(
    url: str,
    pattern: Optional[str] = None,
    limit: int = 10,
    timeout_s: float = 60.0,
    include_nav: bool = True,
    include_footer: bool = True,
) -> List[TextContent]:
    """List links on a page (optionally filtered by regex). Returns JSON as text content."""
    resp = await _fetch_html(url, timeout_s=timeout_s)
    soup = BeautifulSoup(resp.text, "lxml")

    if not include_nav:
        for t in soup.select("nav, aside"):
            t.decompose()
    if not include_footer:
        for t in soup.select("footer"):
            t.decompose()

    links = _extract_links(soup, str(resp.url))

    if pattern:
        try:
            rx = re.compile(pattern, flags=re.I)
            links = [l for l in links if rx.search(l["href"]) or rx.search(l["text"])]
        except re.error as e:
            payload = {"url": str(resp.url), "status": resp.status_code, "error": f"Invalid regex: {e}", "links": []}
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]

    lim = max(1, min(int(limit), 200))
    payload = {"url": str(resp.url), "status": resp.status_code, "links": links[:lim]}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


# ---------------------- Tool 3: fetch_dynamic_page (Playwright) ----------------------
@mcp_server.tool()
async def fetch_dynamic_page(
    url: str,
    wait: str = "networkidle",            # "load" | "domcontentloaded" | "networkidle"
    wait_selector: Optional[str] = None,   # wait for CSS selector if given
    text_only: bool = True,                # True: text; False: HTML
    max_chars: int = 20000,
    scroll: bool = False,                  # auto scroll to bottom
    timeout_s: float = 60.0,               # seconds
    user_agent: Optional[str] = None,      # UA override
) -> List[TextContent]:
    """Render the page with Playwright (Chromium) and return text/HTML and links. Returns JSON as text content."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context_kwargs: Dict[str, Any] = {}
        if user_agent:
            context_kwargs["user_agent"] = user_agent
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until=wait, timeout=int(timeout_s * 1000))

            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=int(timeout_s * 1000))

            if scroll:
                await page.evaluate(
                    """() => new Promise(resolve => {
                        const step = () => {
                            window.scrollBy(0, window.innerHeight);
                            if ((window.innerHeight + window.scrollY) >= document.body.scrollHeight) {
                                resolve();
                            } else {
                                setTimeout(step, 300);
                            }
                        };
                        step();
                    })"""
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

            title = (await page.title()) or ""

            if text_only:
                content = await page.inner_text("body", timeout=5000)
            else:
                content = await page.content()

            # links absolute
            hrefs = await page.eval_on_selector_all(
                "a[href]", "els => els.map(a => ({text: a.innerText, href: a.getAttribute('href')}))"
            )
            links = []
            seen = set()
            for l in hrefs:
                href = (l.get("href") or "").strip()
                if not href:
                    continue
                abs_href = urljoin(url, href)
                if abs_href in seen:
                    continue
                seen.add(abs_href)
                txt = re.sub(r"\s+", " ", (l.get("text") or "").strip())[:200]
                links.append({"text": txt, "href": abs_href})

            if isinstance(content, str) and len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]"

            payload = {"url": url, "title": title, "content": content, "links": links[:200]}
            await context.close(); await browser.close()
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]
        except Exception as e:
            await context.close(); await browser.close()
            payload = {"url": url, "error": f"{type(e).__name__}: {e}"}
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


# ---------------------- entry ----------------------
if __name__ == "__main__":
    # run as stdio MCP server
    mcp_server.run(transport="stdio")
