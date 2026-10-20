"""Explicit web context for Playground turns, using anonymous Parallel Search MCP."""

import json
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "MemoryOS-Playground/1.0 (+https://github.com/BAI-LAB/MemoryOS)"
TIMEOUT_SECONDS = 30
MAX_CONTEXT_CHARS = 25000


class WebContextError(Exception):
    """A requested web action failed; do not silently answer without it."""


@dataclass
class WebContext:
    text: str
    sources: list


def is_web_url(value):
    try:
        url = urlsplit(value)
        return (url.scheme in ("http", "https") and bool(url.hostname)
                and not url.username and not url.password
                and not any(char.isspace() for char in value))
    except ValueError:
        return False


def read_result(result):
    if result.isError:
        raise WebContextError("Web context is unavailable. Please try again later.")
    payload = result.structuredContent
    if payload is None:
        text = "\n".join(block.text for block in result.content if block.type == "text")
        try:
            payload = json.loads(text)
        except (ValueError, TypeError) as error:
            raise WebContextError("The web service returned an invalid response.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise WebContextError("The web service returned an invalid response.")

    sources = []
    excerpts = []
    for row in payload["results"][:10]:
        if not isinstance(row, dict) or not is_web_url(row.get("url", "")):
            continue
        title = str(row.get("title") or row["url"])[:300]
        sources.append({"title": title, "url": row["url"]})
        content = row.get("excerpts") or [row.get("full_content", "")]
        if not isinstance(content, list):
            content = [content]
        excerpts.append(title + "\n" + row["url"] + "\n" + "\n".join(map(str, content)))
    if not sources and payload.get("errors"):
        raise WebContextError("The requested page could not be read.")
    return WebContext("\n\n".join(excerpts)[:MAX_CONTEXT_CHARS] or "No web results found.", sources)


class ParallelWebContext:
    """The client is stateless; the caller owns the conversation identity."""

    async def retrieve(self, message, mode, conversation_id):
        if mode not in ("search", "fetch"):
            raise ValueError("Unknown web action")
        if mode == "fetch" and not is_web_url(message):
            raise ValueError("Read URL requires a single HTTP or HTTPS URL.")
        tool = "web_search" if mode == "search" else "web_fetch"
        arguments = {"session_id": conversation_id}
        if mode == "search":
            arguments.update(objective=message, search_queries=[message])
        else:
            arguments.update(urls=[message])
        try:
            # No environment credentials, cookies or Authorization header are used.
            with anyio.fail_after(TIMEOUT_SECONDS) as deadline:
                async with httpx.AsyncClient(
                    headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS,
                    follow_redirects=False,
                ) as http:
                    async with streamable_http_client(MCP_URL, http_client=http) as (read, write, _):
                        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=TIMEOUT_SECONDS)) as client:
                            await client.initialize()
                            tools = await client.list_tools()
                            if tool not in {item.name for item in tools.tools}:
                                raise WebContextError("The web service does not support this action.")
                            result = await client.call_tool(tool, arguments)
            if deadline.cancel_called:
                raise TimeoutError("Web action exceeded its deadline")
            return read_result(result)
        except WebContextError:
            raise
        except TimeoutError as error:
            raise WebContextError("Web context timed out. Please try again later.") from error
        except Exception as error:
            # CancelledError is a BaseException and propagates to the caller.
            raise WebContextError("Web context is unavailable. Please try again later.") from error
