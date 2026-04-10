from __future__ import annotations

import requests
import xml.etree.ElementTree as ET
import logging

log = logging.getLogger(__name__)

def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers. Returns a summary of findings."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        # ArXiv uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return f"No ArXiv results found for '{query}'."
            
        results = []
        for entry in entries:
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            author_names = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text
            link = entry.find('atom:id', ns).text
            
            results.append(
                f"Title: {title}\n"
                f"Authors: {', '.join(author_names)}\n"
                f"Published: {published}\n"
                f"Link: {link}\n"
                f"Summary: {summary[:300]}...\n"
            )
            
        return "\n---\n".join(results)
        
    except Exception as e:
        log.error("ArXiv search error: %s", e)
        return f"Error searching ArXiv: {e}"
