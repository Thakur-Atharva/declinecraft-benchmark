"""
wiki.py
=======
MediaWiki API access and wikitext cleaning for minecraft.wiki. Everything goes through api.php (direct HTML
and action=raw are blocked by the site's bot protection). Requests back off on HTTP 429.
"""
import json
import os
import re
import time
from typing import List, Optional, Set

import requests


BASE_URL = "https://minecraft.wiki"


HEADERS_TO_REMOVE = [
    r"={2,}\s*Issues\s*={2,}",
    r"={2,}\s*Gallery\s*={2,}",
    r"={2,}\s*Achievements\s*={2,}",
    r"={2,}\s*Advancements\s*={2,}",
    r"={2,}\s*Navigation\s*={2,}",
    r"={2,}\s*Sounds\s*={2,}",
    r"={2,}\s*See also\s*={2,}",
    r"={2,}\s*References\s*={2,}",
    r"={2,}\s*External links\s*={2,}",
    r"={2,}\s*In other media\s*={2,}",
    r"={2,}\s*Data values\s*={2,}",
    r"={2,}\s*Screenshots\s*={2,}",
    r"={2,}\s*History\s*={2,}",
    r"={2,}\s*Mob votes\s*={2,}",
    r"={2,}\s*Upcoming mobs\s*={2,}",
    r"={2,}\s*Joke mobs\s*={2,}",
    r"={2,}\s*Unimplemented mobs\s*={2,}",
    r"={2,}\s*Unimplemented features\s*={2,}",
    r"={2,}\s*April Fools\s*={2,}",
    r"={2,}\s*Trivia\s*={2,}",
    r"={2,}\s*Mentioned\s*={2,}",
    r"={2,}\s*Removed mobs\s*={2,}",
    r"={2,}\s*Removed features\s*={2,}",
]


def wikitext_to_paragraphs(raw_wikitext: str) -> str:
    """
    Cleans raw MediaWiki source (Wikitext) by stripping non-prose elements
    (templates, categories, file embeds, tables, and leakage sections).
    """
    # 1. Target templates with parameters
    text = re.sub(r'\{\{.*?\|([^\{\|\}]+)\}\}', r'\1', raw_wikitext, flags=re.DOTALL)

    # 2. Match and remove <gallery> tags
    text = re.sub(r'<gallery>.*?</gallery>', '', text, flags=re.DOTALL)

    # 3. Match and remove HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', '', text)

    # 4. Targeted Section Removal
    for header_regex in HEADERS_TO_REMOVE:
        pattern = re.compile(rf'^{header_regex}[\s\S]*?(?=\n==|\Z)', re.MULTILINE)
        text = pattern.sub('', text)

    # 5. Interlanguage links
    text = re.sub(r'\[\[[a-z]{2}:.*?\]\]', '', text)

    # 6. Internal links: [[target|display text]] -> display text
    text = re.sub(r'\[\[[^|\]]+?\|([^\]]+?)\]\]', r'\1', text)
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)

    # 7. External links
    text = re.sub(r'\[\w+:\/\/.*? (.*?)\]', r'\1', text)
    text = re.sub(r'\[\w+://.*?\]', r'', text)

    # 8. File/image links
    text = re.sub(r'\[\[(?:File|Image):.*?\]\]', '', text)

    # 9. Bold/italics markers
    text = re.sub(r"'{2,5}", '', text)

    return text.strip()


def _api_get(params: dict, timeout: int = 60, max_attempts: int = 6) -> Optional[dict]:
    """
    Calls the MediaWiki API with retry and exponential backoff.

    The wiki rate-limits sustained crawling with HTTP 429. Without backoff the caller
    cannot distinguish "throttled" from "no results", which silently starves a
    breadth-first crawl of its frontier. Honours Retry-After when present.
    Returns the parsed JSON body, or None if every attempt failed.
    """
    headers = {'User-Agent': 'DeclineCraftResearch/1.0 (academic research benchmark)'}
    backoff = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(f"{BASE_URL}/api.php", params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                if attempt == max_attempts:
                    print(f"  -> Rate limited; {max_attempts} attempts exhausted")
                    return None
                wait = float(r.headers.get('Retry-After') or backoff)
                print(f"  -> HTTP 429, backing off {wait:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_attempts:
                print(f"  -> API request failed after {attempt} attempts: {e}")
                return None
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    return None


def scrape_revision_content(page, url: str, oldid: str) -> Optional[str]:
    """
    Fetches raw Wikitext content of a historical revision using MediaWiki API
    (action=parse&prop=wikitext). Avoids Cloudflare HTTP 403 on direct HTML.
    Retries with backoff on rate limiting.
    """
    data = _api_get({
        'action': 'parse',
        'oldid': oldid,
        'prop': 'wikitext',
        'format': 'json',
    })
    if data is None:
        return None
    content = data.get('parse', {}).get('wikitext', {}).get('*', '')
    return content or None


def find_links_on_page(page, url: str, oldid: str) -> Optional[List[str]]:
    """
    Finds valid wiki article links for a revision using MediaWiki API
    (action=parse&prop=links).

    Returns None if the API call failed (e.g. sustained rate limiting) and an
    empty list only when the revision genuinely has no article links. Callers
    must distinguish the two: treating a throttled fetch as "no links" starves
    a breadth-first crawl and silently truncates the corpus.
    """
    data = _api_get({
        'action': 'parse',
        'oldid': oldid,
        'prop': 'links',
        'format': 'json',
    }, timeout=30)
    if data is None:
        return None

    links = []
    for link in data.get('parse', {}).get('links', []):
        if link.get('ns') != 0:      # ns=0 is the main article namespace
            continue
        title = link.get('*', '')
        if title:
            links.append(f"{BASE_URL}/w/{title.replace(' ', '_')}")
    return links


def load_processed_urls(json_file_path: str) -> Set[str]:
    """Loads URLs from an existing scraped data JSON to prevent duplicate scraping."""
    if not os.path.exists(json_file_path):
        return set()
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed_urls = {item['url'].split('#')[0] for item in data.values() if 'url' in item}
        print(f"Loaded {len(processed_urls)} existing URLs from {json_file_path}.")
        return processed_urls
    except Exception as e:
        print(f"Error loading {json_file_path}: {e}")
        return set()


EXCLUDED_CATEGORY_PREFIXES = (
    "april_fools",
    "joke_",
    "jokes",
)


def get_page_categories(title: str) -> Optional[List[str]]:
    """
    Returns a page's MediaWiki categories, or None if the lookup failed.
    None must not be treated as "no categories": that would admit joke pages.
    """
    data = _api_get({
        'action': 'parse',
        'page': title.replace('_', ' '),
        'prop': 'categories',
        'format': 'json',
    }, timeout=30)
    if data is None:
        return None
    return [c.get('*', '') for c in data.get('parse', {}).get('categories', [])]


def is_excluded_category(categories: List[str]) -> Optional[str]:
    """Returns the offending category if any marks joke/April Fools content."""
    for c in categories or []:
        cl = c.lower().replace("'", "").replace(" ", "_")
        if cl.startswith(EXCLUDED_CATEGORY_PREFIXES):
            return c
    return None


def get_category_members(category: str, limit: int = 500, max_pages: int = 20) -> List[str]:
    """
    Lists main-namespace page titles in a category, following continuation.

    Link-following BFS is seed-limited: it exhausts once the frontier closes,
    even while whole areas of the wiki remain unvisited. Enumerating gameplay
    categories reaches them directly.
    """
    titles: List[str] = []
    cont = None
    for _ in range(max_pages):
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': limit,
            'cmnamespace': 0,
            'format': 'json',
        }
        if cont:
            params['cmcontinue'] = cont
        data = _api_get(params, timeout=30)
        if data is None:
            break
        titles += [m['title'] for m in data.get('query', {}).get('categorymembers', [])]
        cont = data.get('continue', {}).get('cmcontinue')
        if not cont:
            break
    return titles


SEED_CATEGORIES = [
    "Blocks", "Items", "Mobs", "Entities", "Biomes", "Enchantments",
    "Status_effects", "Structures", "Generated_structures", "Commands",
    "Game_terms", "Food", "Tools", "Weapons", "Armor", "Plants",
    "Redstone", "Ores", "Potions", "Villagers", "Transportation",
    "Advancements", "Mechanics", "Non-renewable_resources", "Renewable_resources",
]
