# mcp/web_search_server.py
# A minimal MCP stdio server that exposes a web search tool (DuckDuckGo).

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional

from mcp.server import FastMCP
from ddgs import DDGS

mcp_server = FastMCP(
    name="web_search_mcp_server",
    instructions="DuckDuckGo-based web search tool"
)

def _search_sync(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    time_range: Optional[str] = None,
    backend: str = "html",        # 关键：强制用 DuckDuckGo 的 html 后端
) -> List[Dict[str, Any]]:
    max_results = max(1, min(int(max_results), 50))
    # backend 可选: "html" | "lite" | "api"
    # - html/lite 走 DuckDuckGo 自家页面解析，更稳
    # - api 有时会走到 bing，容易 302/风控
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=time_range,     # "d"/"w"/"m"/"y" 或 None
            max_results=max_results,
            backend=backend,          # <- 这里切换
        )
    out = []
    for r in results:
        out.append({
            "title": r.get("title", ""),
            "url": r.get("href", "") or r.get("url", ""),
            "snippet": r.get("body", "") or r.get("snippet", ""),
        })
    return out

@mcp_server.tool()
async def web_search(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    time_range: Optional[str] = None,
    site: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a web search and return titles/urls/snippets.

    Args:
        query: Search query string.
        max_results: Max number of results (1-50).
        region: DuckDuckGo region (e.g., "us-en", "cn-zh", "jp-ja").
        safesearch: "off" | "moderate" | "strict".
        time_range: None or one of {"d","w","m","y"} for last day/week/month/year.
        site: If provided, search only within the given domain (like "zju.edu.cn").

    Returns:
        {
          "query": str,
          "region": str,
          "safesearch": str,
          "time_range": str|None,
          "results": [{"title": str, "url": str, "snippet": str}, ...]
        }
    """
    q = query.strip()
    if site:
        q = f"site:{site} {q}"

    try:
        results = await asyncio.to_thread(
            _search_sync,
            q, max_results, region, safesearch, time_range
        )
        return {
            "query": q,
            "region": region,
            "safesearch": safesearch,
            "time_range": time_range,
            "results": results,
        }
    except Exception as e:
        return {
            "query": q,
            "error": f"{type(e).__name__}: {e}",
            "results": []
        }

@mcp_server.tool()
async def fetch_dynamic_page(
    url: str,
    wait: str = "networkidle",            # "load" | "domcontentloaded" | "networkidle"
    wait_selector: Optional[str] = None,   # 若提供：等某个 CSS 节点出现
    text_only: bool = True,                # True: 纯文本；False: 返回 HTML
    max_chars: int = 20000,
    scroll: bool = False,                  # 是否滚动到底（适合懒加载）
    timeout_s: float = 20.0,               # 总体超时时间
    user_agent: Optional[str] = None,      # 覆盖 UA
) -> Dict[str, Any]:
    """
    使用无头浏览器渲染后抓取页面内容（适合 JS 动态渲染页面）。

    Args:
        url: 目标 URL
        wait: 页面等待条件: "load"/"domcontentloaded"/"networkidle"
        wait_selector: 若提供，则额外等待该 CSS 选择器出现
        text_only: True 返回纯文本，False 返回 HTML
        max_chars: 内容截断长度
        scroll: 是否自动滚动到底（触发懒加载）
        timeout_s: 超时秒数
        user_agent: 可选自定义 UA

    Returns:
        {url, title, content, links}
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context_kwargs = {}
        if user_agent:
            context_kwargs["user_agent"] = user_agent
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until=wait, timeout=int(timeout_s * 1000))

            # 可选等待指定节点
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=int(timeout_s * 1000))

            # 可选滚动触发懒加载
            if scroll:
                # 简单滚动到底；可按需改进为分段滚动
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
                # 滚动后再等一小会儿网络空闲
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass

            # 标题
            title = (await page.title()) or ""

            # 内容
            if text_only:
                # 纯文本
                content = await page.inner_text("body", timeout=5000)
            else:
                content = await page.content()

            # 链接（绝对化）
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
                text = re.sub(r"\s+", " ", (l.get("text") or "").strip())[:200]
                links.append({"text": text, "href": abs_href})

            # 截断
            if isinstance(content, str) and len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]"

            await context.close()
            await browser.close()

            return {
                "url": url,
                "title": title,
                "content": content,
                "links": links[:200],
            }
        except Exception as e:
            await context.close()
            await browser.close()
            return {"url": url, "error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    mcp_server.run()
