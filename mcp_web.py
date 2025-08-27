# mcp_web.py
from fastapi import FastAPI
import httpx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Web Search")

DUCKDUCKGO_API = "https://api.duckduckgo.com/"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MCP Web Search Server",
        "features": ["DuckDuckGo Search"],
        "async": True,
        "version": "2.0.0"
    }

@app.get("/search")
async def search_web(query: str, max_results: int = 5):
    """
    Web search using DuckDuckGo Instant Answer API (async)
    """
    try:
        logger.info(f"Performing web search: {query}")
        
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(DUCKDUCKGO_API, params=params)
            data = response.json()

        results = []
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text"),
                    "url": topic.get("FirstURL")
                })

        logger.info(f"Search completed, found {len(results)} results")
        return {"query": query, "results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"error": f"Search failed: {e}", "query": query, "results": []}

@app.get("/info")
async def get_info():
    """Get server information"""
    return {
        "title": "MCP Web Search Server",
        "description": "Web search server using DuckDuckGo API",
        "version": "2.0.0",
        "endpoints": {
            "/search": "Perform web search",
            "/health": "Health check",
            "/info": "Server information"
        },
        "async_compatible": True
    }