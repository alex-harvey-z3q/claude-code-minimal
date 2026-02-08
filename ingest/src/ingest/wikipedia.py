import requests

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "wiki-rag-ingest/0.1 (contact: alexharv074@gmail.com)"

def fetch_page(title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    resp = requests.get(API_URL, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    return next(iter(pages.values()))
