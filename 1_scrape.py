#!/usr/bin/env python
"""
1_scrape.py
===========
Builds the two corpora from minecraft.wiki through the MediaWiki API.

  list    the complete page list: every current article of the main and Tutorial namespaces with its current
          categories, and every redirect last edited after the cutoff (it may have been an article then).
  pre     C_pre: every listed page at its newest revision at or before the cutoff (31 March 2025). Version and
          changelog pages, mob-vote proposals, redirects and pages under 200 characters are skipped.
  post    C_post: every listed article whose FIRST revision is after the cutoff (pages that existed before and were
          later updated are excluded), at its latest revision.
  filter  drops pages that are not real, current base-game mechanics: joke/April Fools, version, story and spin-off,
          media and merchandise, creator docs, non-article pages, structure blueprints, soundtrack tracks, maps,
          franchise and novel pages, protocol and dated pages (both corpora); set-index pages, planned features and
          tutorials (C_post only: post-cutoff tutorials are guides to old mechanics).

Every API failure raises and is retried on the next pass, never read as "no revision" or "no categories": under the
wiki's rate limiting, a throttled request otherwise drops real pages (Drowned, Phantom, Name Tag, ...). Each stage
resumes if rerun, and exits non-zero while pages are still failing.

    python 1_scrape.py list
    python 1_scrape.py pre
    python 1_scrape.py post
    python 1_scrape.py filter

The paper's corpora were built by link-following crawls with page caps, which missed about 900 articles that
existed at the cutoff; they are kept as `crawl-pre` and `crawl-post`.
"""
import argparse
import json
import os
import random
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from config import CUT_OFF_DATE, TEST_JSON_FILE, TRAIN_JSON_FILE
from wiki import (BASE_URL, SEED_CATEGORIES, _api_get, find_links_on_page, get_category_members,
                  get_page_categories, is_excluded_category, load_processed_urls, scrape_revision_content,
                  wikitext_to_paragraphs)

TARGET_DATE = CUT_OFF_DATE
HEADERS = {'User-Agent': 'DeclineCraftResearch/1.0 (academic research benchmark)'}


# --- C_pre: the newest revision of each page at or before the cutoff -------------------------------


BLACKLIST_KEYWORDS = [
    "minecraft_live", "fandom", "movies", "live_event", "vote_event",
    "skin_pack", "minecon", "planned_versions", "winter_drop",
    "java_edition_", "bedrock_edition_",
    # Announced but never implemented.
    "mentioned_features",
]


VOTE_PROPOSAL_RE = re.compile(
    r"\b(was|is) (a )?(proposed|candidate|planned) (mob|biome|feature)\b|\bwas a mob (that (was )?(proposed|featured)|planned)\b"
    r"|\bfeatured in the (mob|biome) vote\b|\blost (the|in the) (mob|biome) vote\b", re.IGNORECASE)


VERSION_PAGE_RE = re.compile(
    r"^(?:\d+\.\d+|\d{2}w\d{2}[a-z]?|v?\d+\.\d+(?:\.\d+)?(?:[-_](?:pre|rc|alpha|beta|snapshot|build))?)",
    re.IGNORECASE,
)


VERSION_TOKENS = ("_alpha", "_beta", "_build_", "_pre-release", "_snapshot", "_release_candidate")


MIN_CONTENT_CHARS = 200


def _skip_path(output_path: str) -> str:
    """Sidecar file recording URLs fetched but not stored (redirects, stubs)."""
    return f"{output_path}.skipped.json"


def _load_skipped(output_path: str) -> Set[str]:
    path = _skip_path(output_path)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_skipped(output_path: str, urls: Set[str]) -> None:
    try:
        with open(_skip_path(output_path), 'w', encoding='utf-8') as f:
            json.dump(sorted(urls), f)
    except Exception as e:
        print(f"  -> Could not write skipped-URL sidecar: {e}")


def is_blacklisted(url: str) -> bool:
    """True if the URL is an out-of-scope topic or a version/changelog page."""
    url_lower = url.lower()
    if any(keyword in url_lower for keyword in BLACKLIST_KEYWORDS):
        return True

    title = url.split('/w/')[-1]
    title_lower = title.lower()
    if VERSION_PAGE_RE.match(title) or any(t in title_lower for t in VERSION_TOKENS):
        return True
    return False


def is_redirect(wikitext: str) -> bool:
    """True if the wikitext is a redirect stub rather than an article."""
    return wikitext.lstrip().lower().startswith(("#redirect", "#weiterleitung"))


def get_latest_revision_before_date(
    url: str,
    target_date: datetime = TARGET_DATE,
) -> Optional[Tuple[str, datetime]]:
    """
    Finds the oldid and timestamp of the most recent revision predating target_date.

    Uses action=query&prop=revisions&rvdir=older&rvstart=<cutoff>, which returns the
    newest revision at or before the cutoff in a single request.
    """
    title = url.split('/w/')[-1]
    target_dt = target_date if target_date.tzinfo else target_date.replace(tzinfo=timezone.utc)
    target_iso = target_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        r = requests.get(
            f"{BASE_URL}/api.php",
            params={
                'action': 'query',
                'prop': 'revisions',
                'titles': title,
                'rvlimit': 1,
                'rvstart': target_iso,
                'rvdir': 'older',
                'rvprop': 'ids|timestamp',
                'format': 'json',
            },
            headers=HEADERS,
            timeout=30,
        )
        if r.status_code != 200:
            return None
        pages = r.json().get('query', {}).get('pages', {})
        for pid, pdata in pages.items():
            if pid == '-1' or pdata.get('missing') is True:
                return None
            revs = pdata.get('revisions', [])
            if revs:
                oldid = str(revs[0].get('revid'))
                ts = revs[0].get('timestamp')
                if oldid and ts:
                    return oldid, datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
        return None
    except Exception:
        return None


def _get_links(url: str, oldid: str, crawl_from_links: bool) -> Tuple[List[str], bool]:
    """Returns (links, fetch_failed). fetch_failed distinguishes throttling from no links."""
    if not crawl_from_links:
        return [], False
    links = find_links_on_page(None, url, oldid)
    if links is None:
        return [], True
    return [ln.split('#')[0] for ln in links], False


def _fetch_one(url: str, crawl_from_links: bool) -> Optional[Dict[str, Any]]:
    """
    Resolves the pre-cutoff revision for a single page and fetches its wikitext.
    Returns a record dict, or None if no suitable revision exists.
    Runs inside the thread pool; performs no shared-state mutation.
    """
    revision_info = get_latest_revision_before_date(url, TARGET_DATE)
    if not revision_info:
        return None
    oldid, revision_date = revision_info

    wikitext = scrape_revision_content(None, url, oldid)
    if not wikitext:
        return None

    title = url.split('/w/')[-1]
    links, link_failed = _get_links(url, oldid, crawl_from_links)

    # Redirect stubs ("#redirect Effect") carry no mechanics. Their link targets are
    # still returned so the crawler reaches the real article.
    if is_redirect(wikitext):
        return {'page_title': title, 'url': url, 'skipped': 'redirect',
                'links': links, 'link_failed': link_failed}

    clean_content = wikitext_to_paragraphs(wikitext)
    if not clean_content or len(clean_content.strip()) < MIN_CONTENT_CHARS:
        return {'page_title': title, 'url': url, 'skipped': 'too_short',
                'links': links, 'link_failed': link_failed}
    if VOTE_PROPOSAL_RE.search(clean_content[:1500]):
        return {'page_title': title, 'url': url, 'skipped': 'vote_proposal',
                'links': links, 'link_failed': link_failed}

    return {
        'page_title': title,
        'url': url,
        'oldid': oldid,
        'date': revision_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'scraped_at': datetime.now().isoformat(),
        'content': clean_content,
        'links': links,
        'link_failed': link_failed,
    }


def scrape_wiki_pre_cutoff(
    start_urls: Optional[List[str]] = None,
    output_path: str = TRAIN_JSON_FILE,
    max_pages: int = 2000,
    concurrency: int = 4,
    crawl_from_links: bool = True,
    seed: int = 42,
    flush_every: int = 25,
    seed_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Crawls the wiki breadth-first, storing each page's newest revision predating
    CUT_OFF_DATE. Resumable: an existing output file is loaded and extended.
    """
    random.seed(seed)

    if start_urls is None:
        start_urls = list(DEFAULT_SEED_URLS)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    scraped_data: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            print(f"Resuming with {len(scraped_data)} already scraped pages in {output_path}")
        except Exception:
            scraped_data = {}

    processed_urls: Set[str] = load_processed_urls(output_path)
    # Redirects and stubs are fetched but never stored, so they are absent from the
    # corpus and would be re-fetched on every resume. The sidecar records them.
    skipped_urls: Set[str] = _load_skipped(output_path)
    if skipped_urls:
        print(f"Skipping {len(skipped_urls)} URLs previously fetched and filtered.")
    queued: Set[str] = set(processed_urls) | skipped_urls

    frontier: List[str] = []
    for u in start_urls:
        base = u.split('#')[0]
        if base not in queued and not is_blacklisted(base):
            frontier.append(base)
            queued.add(base)

    # Resuming: the frontier is not persisted, and every seed is already in
    # processed_urls, so rebuild the frontier from the links of pages already
    # scraped. Without this a resumed run starts with an empty frontier and
    # exits immediately, silently re-doing nothing.
    if scraped_data:
        recovered = 0
        for rec in scraped_data.values():
            for link in (rec.get('links') or []):
                link = link.split('#')[0]
                if link not in queued and not is_blacklisted(link):
                    frontier.append(link)
                    queued.add(link)
                    recovered += 1
        print(f"Resume: rebuilt frontier with {recovered} unvisited links from {len(scraped_data)} stored pages.")

    # Category seeding reaches pages that link-following never gets to: BFS closes
    # once the frontier empties, which happened at 3,823 pages while hundreds of
    # gameplay pages remained unvisited.
    if seed_categories:
        print(f"Enumerating {len(seed_categories)} gameplay categories for seeds...")
        added = 0
        for cat in seed_categories:
            for title in get_category_members(cat):
                url = f"{BASE_URL}/w/{title.replace(' ', '_')}"
                if url not in queued and not is_blacklisted(url):
                    frontier.append(url)
                    queued.add(url)
                    added += 1
        print(f"Category seeding added {added} candidate pages.")

    print(f"Starting pre-cutoff crawler: {len(frontier)} seeds, target {max_pages} pages, "
          f"{concurrency} workers, cutoff {TARGET_DATE.isoformat()}")

    since_flush = 0
    throttled = 0
    skipped_counts: Dict[str, int] = {}
    while frontier and len(scraped_data) < max_pages:
        remaining = max_pages - len(scraped_data)
        batch_size = min(max(concurrency * 4, 16), len(frontier), max(remaining, 1) * 2)
        current_batch = [frontier.pop(0) for _ in range(batch_size)]

        records: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_fetch_one, url, crawl_from_links): url
                for url in current_batch
            }
            for future in as_completed(futures):
                url = futures[future]
                processed_urls.add(url)
                try:
                    rec = future.result()
                except Exception as e:
                    print(f"  -> Error on {url}: {e}")
                    continue
                if rec:
                    records.append(rec)

        # Sort so insertion order and frontier expansion are scheduling-independent.
        records.sort(key=lambda r: r['page_title'])

        new_links: List[str] = []
        requeue: List[str] = []
        for rec in records:
            title = rec.pop('page_title')
            # A throttled link fetch is not the same as a page with no links. Put the
            # page back on the frontier so its links are collected on a later pass.
            if rec.pop('link_failed', False):
                requeue.append(rec['url'])
                throttled += 1
            # Every fetched page expands the frontier, including redirects, stubs and
            # pages already stored from an earlier pass -- harvest links before any
            # storage checks, or a re-queued page contributes nothing on retry.
            new_links.extend(rec['links'])

            if rec.get('skipped'):
                skipped_counts[rec['skipped']] = skipped_counts.get(rec['skipped'], 0) + 1
                skipped_urls.add(rec['url'])
                continue
            if title in scraped_data:
                continue
            if len(scraped_data) >= max_pages:
                continue
            scraped_data[title] = rec
            since_flush += 1
            print(f"  [{len(scraped_data)}/{max_pages}] {title} (rev {rec['oldid']} @ {rec['date']})")

        # Preserve natural link order rather than sorting alphabetically. Sorted
        # insertion into a FIFO frontier makes the crawl march A->Z and strand whole
        # topic areas; records are already sorted, so order stays deterministic.
        seen_in_batch = set()
        for link in new_links:
            if link in seen_in_batch or link in queued or is_blacklisted(link):
                continue
            seen_in_batch.add(link)
            frontier.append(link)
            queued.add(link)

        for url in requeue:
            queued.discard(url)
            frontier.append(url)

        if since_flush >= flush_every:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False)
            _save_skipped(output_path, skipped_urls)
            print(f"  -> Flushed {len(scraped_data)} pages to {output_path}")
            since_flush = 0

        # Politeness pause between batches.
        time.sleep(random.uniform(0.2, 0.6))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    _save_skipped(output_path, skipped_urls)

    print(f"\nFinished pre-cutoff scraping. Total saved: {len(scraped_data)} -> {output_path}")
    if throttled:
        print(f"Link fetches throttled and re-queued: {throttled}")
    if skipped_counts:
        print(f"Filtered during crawl: {skipped_counts} "
              f"(redirect stubs and sub-{MIN_CONTENT_CHARS}-char pages are crawled for links but not indexed)")
    return scraped_data


START_PAGES = [
    "https://minecraft.wiki/w/Trading",
    "https://minecraft.wiki/w/Brewing",
    "https://minecraft.wiki/w/Enchanting",
    "https://minecraft.wiki/w/Crafting",
    "https://minecraft.wiki/w/Smelting",
    "https://minecraft.wiki/w/Mob",
    "https://minecraft.wiki/w/Item",
    "https://minecraft.wiki/w/Block",
    "https://minecraft.wiki/w/Biome",
    "https://minecraft.wiki/w/Redstone_circuits",
    "https://minecraft.wiki/w/Effects",
    "https://minecraft.wiki/w/Smithing",
    "https://minecraft.wiki/w/Structure",
    "https://minecraft.wiki/w/Tutorials",
    "https://minecraft.wiki/w/Java_Edition_guides",
    "https://minecraft.wiki/w/Java_Edition_guides/Caves_%26_Cliffs:_Part_II",
    "https://minecraft.wiki/w/Java_Edition_guides/Caves_%26_Cliffs:_Part_I",
    "https://minecraft.wiki/w/Tutorial:Combat",
    "https://minecraft.wiki/w/Tutorial:Mining",
    "https://minecraft.wiki/w/Tutorial:Quick_ways_to_get_food",
    "https://minecraft.wiki/w/Tutorial:Breaking_a_fall",
    "https://minecraft.wiki/w/Tutorial:Advancement_guide",
    "https://minecraft.wiki/w/Farming",
]


# --- C_post: pages whose first revision is after the cutoff ----------------------------------------


DISALLOWED_PREFIXES = (
    'Pocket Edition', 'Java Edition 1.', 'Java Edition Alpha', 'Java Edition Beta',
    'Bedrock Edition 1.', 'Bedrock Edition Alpha', 'Bedrock Edition Beta',
    'Console Edition', 'Alpha ', 'Beta ', 'Classic ', 'Indev ', 'Infdev ',
    'PlayStation', 'Xbox', 'Nintendo', 'Wii', 'MinecraftEdu', 'Pi Edition',
    'Minecraft China', 'Special:', 'Category:', 'Template:', 'Talk:', 'User:', 'File:',
    'Help:', 'Minecraft Wiki:', 'Module:', 'MediaWiki:', 'Draft:'
)


DISALLOWED_SUBSTRINGS = [
    'preview', 'snapshot', 'release candidate', 'pre-release', 'skin pack',
    'minecon', 'minecraft live', 'trailer', 'announcement', 'lego', 'dungeons:',
    'legends:', 'mumbo jumbo', 'tubbo', 'experience:', 'soundtrack', 'album',
    'commands/', 'tutorials/', 'programs/', 'guides/'
]


def is_candidate_article(title: str) -> bool:
    """Returns True if title represents potential in-game content."""
    if not title or len(title.strip()) < 2:
        return False
    title_clean = title.strip()
    if any(title_clean.startswith(p) for p in DISALLOWED_PREFIXES):
        return False
    lower = title_clean.lower()
    if any(s in lower for s in DISALLOWED_SUBSTRINGS):
        return False
    return True


def check_oldest_revision(title: str, target_date: datetime = TARGET_DATE) -> Tuple[str, Optional[str], bool]:
    """
    Checks if a page was created strictly on or after target_date.
    Returns (title, oldest_timestamp, is_post_cutoff).
    """
    data = _api_get({
        'action': 'query',
        'prop': 'revisions',
        'titles': title,
        'rvlimit': 1,
        'rvdir': 'newer',
        'rvprop': 'ids|timestamp',
        'format': 'json',
    }, timeout=30)
    if data is None:
        return title, None, False
    try:
        pages = data.get('query', {}).get('pages', {})
        for pid, pdata in pages.items():
            if pid == '-1' or pdata.get('missing') is True:
                return title, None, False
            revs = pdata.get('revisions', [])
            if revs:
                ts = revs[0].get('timestamp')
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                target_dt = target_date if target_date.tzinfo else target_date.replace(tzinfo=timezone.utc)
                return title, ts, (dt >= target_dt)
        return title, None, False
    except Exception:
        return title, None, False


def get_latest_revision_info(title: str) -> Optional[Tuple[str, str]]:
    """Gets the latest revision ID (oldid) and timestamp for a title."""
    data = _api_get({
        'action': 'query',
        'prop': 'revisions',
        'titles': title,
        'rvlimit': 1,
        'rvprop': 'ids|timestamp',
        'format': 'json',
    }, timeout=30)
    if data is None:
        return None
    try:
        pages = data.get('query', {}).get('pages', {})
        for pid, pdata in pages.items():
            revs = pdata.get('revisions', [])
            if revs:
                return str(revs[0].get('revid')), revs[0].get('timestamp')
        return None
    except Exception:
        return None


def get_article_links(title: str) -> List[str]:
    """
    Retrieves outgoing main-namespace article links from a page.
    Returns [] on a genuinely link-free page; a throttled fetch is retried by
    _api_get rather than being silently reported as "no links".
    """
    data = _api_get({
        'action': 'parse',
        'page': title,
        'prop': 'links',
        'format': 'json',
    }, timeout=30)
    if data is None:
        return []
    raw_links = data.get('parse', {}).get('links', [])
    return [l['*'] for l in raw_links if l.get('ns') == 0 and is_candidate_article(l.get('*', ''))]


def scrape_wiki_post_cutoff(
    seed_titles: Optional[List[str]] = None,
    output_path: str = TEST_JSON_FILE,
    max_pages: int = 300,
    concurrency: int = 8,
    use_category_seeds: bool = False,
):
    """Crawls wiki pages, collecting pages created strictly on or after CUT_OFF_DATE."""
    if seed_titles is None:
        seed_titles = [
            "The Copper Age", "Mounts of Mayhem", "Tiny Takeover", "Chaos Cubed", "Chase the Skies",
            "Copper axe", "Copper pickaxe", "Copper shovel", "Copper hoe", "Copper sword",
            "Copper helmet", "Copper chestplate", "Copper leggings", "Copper boots", "Copper horse armor",
            "Copper nugget", "Sulfur pool", "Sulfur caves", "Polished sulfur wall", "Polished cinnabar wall",
            "Cinnabar slab", "Cinnabar stairs", "Cinnabar brick slab", "Cinnabar Brick Wall", "Baby nautilus",
            "Golden dandelion", "Straw Bed", "Dappled Forest", "Wool Slab", "Concrete Slab", "Concrete Stairs",
            "Poplar Planks", "Stripped Poplar Wood", "Red Shrub", "Baby cat"
        ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scraped_data: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            print(f"Resuming with {len(scraped_data)} already scraped pages in {output_path}")
        except Exception:
            scraped_data = {}

    skip_titles: Set[str] = set()
    for existing_file in [TRAIN_JSON_FILE]:
        if os.path.exists(existing_file):
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                for k in d.keys():
                    skip_titles.add(k.replace('_', ' '))
            except Exception:
                pass

    visited_or_queued: Set[str] = set(skip_titles)
    for k in scraped_data.keys():
        visited_or_queued.add(k.replace('_', ' '))

    candidates_queue: List[str] = [s for s in seed_titles if s not in visited_or_queued and is_candidate_article(s)]
    for s in candidates_queue:
        visited_or_queued.add(s)

    # Resuming: the queue is not persisted and every seed is already scraped, so
    # rebuild from the links of stored pages. Without this a resumed run starts
    # with an empty queue and exits immediately having done nothing.
    if scraped_data:
        recovered = 0
        for rec in scraped_data.values():
            for link in (rec.get('links') or []):
                t = link.split('/w/')[-1].replace('_', ' ')
                if t and t not in visited_or_queued and is_candidate_article(t):
                    candidates_queue.append(t)
                    visited_or_queued.add(t)
                    recovered += 1
        print(f"Resume: rebuilt queue with {recovered} unvisited candidates from {len(scraped_data)} stored pages.")

    # Gameplay categories reach post-cutoff pages that link-following misses.
    if use_category_seeds:
        added = 0
        for cat in SEED_CATEGORIES:
            for t in get_category_members(cat):
                if t not in visited_or_queued and is_candidate_article(t):
                    candidates_queue.append(t)
                    visited_or_queued.add(t)
                    added += 1
        print(f"Category seeding added {added} candidates.")

    skipped: Dict[str, int] = {}
    print(f"Starting post-cutoff crawler with {len(candidates_queue)} initial candidates...")

    while candidates_queue and len(scraped_data) < max_pages:
        batch_size = min(30, len(candidates_queue))
        current_batch = [candidates_queue.pop(0) for _ in range(batch_size)]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(check_oldest_revision, title, TARGET_DATE): title for title in current_batch}
            verified_post: List[Tuple[str, str]] = []
            for future in as_completed(futures):
                title, ts, is_post = future.result()
                if is_post and ts:
                    verified_post.append((title, ts))

        for title, earliest_ts in verified_post:
            if len(scraped_data) >= max_pages:
                break
            page_key = title.replace(' ', '_')
            if page_key in scraped_data:
                continue

            rev_info = get_latest_revision_info(title)
            if not rev_info:
                continue
            oldid, latest_ts = rev_info

            wikitext = scrape_revision_content(None, f"{BASE_URL}/w/{page_key}", oldid)
            if not wikitext:
                continue

            # Expand the frontier before any content filtering: redirects and stubs
            # are not worth indexing but their links still lead to real articles.
            outbound_links = get_article_links(title)
            for l in outbound_links:
                if l not in visited_or_queued and is_candidate_article(l):
                    visited_or_queued.add(l)
                    candidates_queue.append(l)

            if wikitext.lstrip().lower().startswith(('#redirect', '#weiterleitung')):
                skipped['redirect'] = skipped.get('redirect', 0) + 1
                continue

            clean_content = wikitext_to_paragraphs(wikitext)
            if not clean_content or len(clean_content.strip()) < MIN_CONTENT_CHARS:
                skipped['too_short'] = skipped.get('too_short', 0) + 1
                continue

            canonical_url = f"{BASE_URL}/w/{page_key}"
            scraped_data[page_key] = {
                'url': canonical_url,
                'oldid': oldid,
                'date': latest_ts,
                'earliest_date': earliest_ts,
                'scraped_at': datetime.now().isoformat(),
                'content': clean_content,
                'links': [f"{BASE_URL}/w/{l.replace(' ', '_')}" for l in outbound_links]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=4, ensure_ascii=False)

            print(f"  [SAVED {len(scraped_data)}/{max_pages}] '{page_key}' ({len(clean_content)} chars)")
            time.sleep(0.1)

    print(f"\nFinished post-cutoff scraping. Total saved: {len(scraped_data)}")
    if skipped:
        print(f"Filtered during crawl: {skipped} "
              f"(redirects and sub-{MIN_CONTENT_CHARS}-char stubs crawled for links but not indexed)")


# --- Full page-list crawl (default) ------------------------------------------------------------------------


TUTORIAL_NS = 10010
PAGE_LIST_FILE = os.path.join(os.path.dirname(TRAIN_JSON_FILE) or ".", "page_list.json")
CUTOFF_ISO = TARGET_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')


class WikiAPIError(RuntimeError):
    """An API request failed after its retries. Never to be read as an empty answer."""


def _api(params: dict) -> dict:
    d = _api_get({**params, 'format': 'json'}, timeout=30, max_attempts=10)
    if d is None:
        raise WikiAPIError(f"API failed: {params}")
    return d


def _key(title: str) -> str:
    return title.replace(' ', '_')


def _save_json(obj: Any, path: str) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _all_pages(ns: int, redirects: bool) -> List[str]:
    out, cont = [], {}
    while True:
        d = _api({'action': 'query', 'list': 'allpages', 'apnamespace': ns, 'aplimit': 500,
                  'apfilterredir': 'redirects' if redirects else 'nonredirects', **cont})
        out += [p['title'] for p in d['query']['allpages']]
        if 'continue' not in d:
            return out
        cont = d['continue']


def _query_titles(titles: List[str], params: dict) -> Dict[str, Dict[str, Any]]:
    """prop queries for titles, 50 per request, following continuation: {title: page}."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(titles), 50):
        cont = {}
        while True:
            d = _api({'action': 'query', 'titles': '|'.join(titles[i:i + 50]), **params, **cont})
            for p in d.get('query', {}).get('pages', {}).values():
                cur = out.setdefault(p['title'], {'categories': [], 'revisions': []})
                cur['categories'] += [c['title'].split(':', 1)[1].replace(' ', '_') for c in p.get('categories', [])]
                cur['revisions'] += p.get('revisions', [])
            if 'continue' not in d:
                break
            cont = d['continue']
    return out


def list_pages(path: Optional[str] = None) -> None:
    """The complete page list, with each article's current categories (used by the filter)."""
    path = path or PAGE_LIST_FILE
    articles, late_redirects = {}, []
    for ns in (0, TUTORIAL_NS):
        titles = _all_pages(ns, redirects=False)
        cats = _query_titles(titles, {'prop': 'categories', 'cllimit': 'max'})
        missing = [t for t in titles if t not in cats]
        if missing:
            raise WikiAPIError(f"no category answer for {len(missing)} pages, e.g. {missing[:3]}")
        articles.update({t: cats[t]['categories'] for t in titles})
        redirects = [t for t in _all_pages(ns, redirects=True) if not title_reason(_key(t))]
        revs = _query_titles(redirects, {'prop': 'revisions', 'rvprop': 'timestamp'})
        late_redirects += [t for t, p in revs.items() if p['revisions'] and p['revisions'][0]['timestamp'] > CUTOFF_ISO]
        print(f"namespace {ns}: {len(titles)} articles, {len(redirects)} redirects", flush=True)
    _save_json({'cutoff': CUTOFF_ISO, 'listed_at': datetime.now().isoformat(), 'articles': articles,
                'redirects_after_cutoff': sorted(late_redirects)}, path)
    print(f"{len(articles)} articles and {len(late_redirects)} redirects edited after the cutoff -> {path}")


def _load_page_list(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or PAGE_LIST_FILE
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run `1_scrape.py list` first")
    return json.load(open(path, encoding='utf-8'))


def cutoff_revision(title: str) -> Optional[Tuple[str, datetime]]:
    """(oldid, timestamp) of the newest revision at or before the cutoff, None if the page has none. Raises on API
    failure (get_latest_revision_before_date reads any failure as None)."""
    d = _api({'action': 'query', 'prop': 'revisions', 'titles': title.replace('_', ' '), 'rvlimit': 1,
              'rvstart': CUTOFF_ISO, 'rvdir': 'older', 'rvprop': 'ids|timestamp'})
    for pid, p in d.get('query', {}).get('pages', {}).items():
        if pid == '-1' or 'missing' in p:
            return None
        revs = p.get('revisions') or []
        if revs:
            return str(revs[0]['revid']), datetime.strptime(revs[0]['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    return None


def first_revision(title: str) -> Optional[str]:
    """Timestamp of a page's first revision (page moves keep their history), None if it has none. Raises on API
    failure (check_oldest_revision reads a failure as "not after the cutoff")."""
    d = _api({'action': 'query', 'prop': 'revisions', 'titles': title.replace('_', ' '), 'rvlimit': 1,
              'rvdir': 'newer', 'rvprop': 'timestamp'})
    for pid, p in d.get('query', {}).get('pages', {}).items():
        revs = p.get('revisions') or []
        return None if pid == '-1' or not revs else revs[0]['timestamp']
    return None


def fetch_pre_page(title: str) -> Dict[str, Any]:
    """C_pre record of one page, or {'skipped': reason}. Raises on API failure."""
    rev = cutoff_revision(title)
    if rev is None:
        return {'skipped': 'no revision at or before the cutoff'}
    oldid, revision_date = rev
    url = f"{BASE_URL}/w/{_key(title)}"
    wikitext = scrape_revision_content(None, url, oldid)
    if wikitext is None:
        raise WikiAPIError(f"content fetch failed for {title} ({oldid})")
    if is_redirect(wikitext):
        return {'skipped': 'redirect'}
    clean = wikitext_to_paragraphs(wikitext)
    if not clean or len(clean.strip()) < MIN_CONTENT_CHARS:
        return {'skipped': 'too_short'}
    if VOTE_PROPOSAL_RE.search(clean[:1500]):
        return {'skipped': 'vote_proposal'}
    return {'url': url, 'oldid': oldid, 'date': revision_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'scraped_at': datetime.now().isoformat(), 'content': clean, 'links': []}


def fetch_post_page(title: str, earliest: str) -> Optional[Dict[str, Any]]:
    """C_post record of one page at its latest revision, or None for a redirect or stub. Raises on API failure."""
    info = get_latest_revision_info(title.replace('_', ' '))
    if not info:
        raise WikiAPIError(f"no latest revision for {title}")
    oldid, ts = info
    url = f"{BASE_URL}/w/{_key(title)}"
    wikitext = scrape_revision_content(None, url, oldid)
    if wikitext is None:
        raise WikiAPIError(f"content fetch failed for {title} ({oldid})")
    if is_redirect(wikitext):
        return None
    clean = wikitext_to_paragraphs(wikitext)
    if not clean or len(clean.strip()) < MIN_CONTENT_CHARS:
        return None
    return {'url': url, 'oldid': oldid, 'date': ts, 'earliest_date': earliest,
            'scraped_at': datetime.now().isoformat(), 'content': clean, 'links': []}


def _run_passes(todo: List[str], fetch, record, workers: int, passes: int, label: str) -> List[str]:
    """Calls fetch on every title, retrying failed ones in later passes; record(title, result) stores a result.
    Returns the titles still failing."""
    for n in range(1, passes + 1):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch, t): t for t in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    record(t, fut.result())
                except WikiAPIError:
                    failed.append(t)
                if i % 500 == 0:
                    print(f"  [{label} pass {n}] {i}/{len(todo)}, {len(failed)} failed", flush=True)
        print(f"[{label}] pass {n}: {len(todo) - len(failed)} done, {len(failed)} failed", flush=True)
        if not failed:
            return []
        todo = sorted(failed)
        time.sleep(30 * n)
    return todo


def scrape_pre_full(output_path: str = TRAIN_JSON_FILE, workers: int = 8, passes: int = 4) -> List[str]:
    """C_pre from the page list: every article and every redirect edited after the cutoff, at the cutoff."""
    plist = _load_page_list()
    cands = sorted({_key(t) for t in list(plist['articles']) + plist['redirects_after_cutoff']
                    if not title_reason(_key(t))})
    data = json.load(open(output_path, encoding='utf-8')) if os.path.exists(output_path) else {}
    skip_path = _skip_path(output_path)
    skipped = json.load(open(skip_path, encoding='utf-8')) if os.path.exists(skip_path) else {}
    if not isinstance(skipped, dict):
        skipped = {}
    todo = [t for t in cands if t not in data and t not in skipped]
    print(f"C_pre: {len(cands)} listed pages, {len(data)} stored, {len(skipped)} skipped, {len(todo)} to fetch")
    lock = threading.Lock()

    def record(t, rec):
        with lock:
            if rec.get('skipped'):
                skipped[t] = rec['skipped']
            else:
                data[t] = rec
            if (len(data) + len(skipped)) % 500 == 0:
                _save_json(data, output_path)
                _save_json(skipped, skip_path)

    failing = _run_passes(todo, fetch_pre_page, record, workers, passes, 'C_pre')
    _save_json(data, output_path)
    _save_json(skipped, skip_path)
    print(f"C_pre: {len(data)} pages -> {output_path}; skipped {dict(Counter(skipped.values()))}")
    return failing


def scrape_post_full(output_path: str = TEST_JSON_FILE, workers: int = 8, passes: int = 4) -> List[str]:
    """C_post from the page list: every main-namespace article whose first revision is after the cutoff. Titles the
    filter would reject by title or current category are not fetched."""
    plist = _load_page_list()
    cands = sorted(t for t, cats in plist['articles'].items()
                   if not t.startswith('Tutorial:') and is_candidate_article(t)
                   and not classify((_key(t), {'categories': cats}), False)[1])
    first_path = f"{output_path}.first_revisions.json"
    first = json.load(open(first_path, encoding='utf-8')) if os.path.exists(first_path) else {}
    lock = threading.Lock()

    def record_first(t, ts):
        with lock:
            first[t] = ts

    failing = _run_passes([t for t in cands if t not in first], first_revision, record_first, workers, passes,
                          'first revisions')
    _save_json(first, first_path)
    target = datetime.strptime(CUTOFF_ISO, '%Y-%m-%dT%H:%M:%SZ')
    post = sorted(t for t in cands if first.get(t) and datetime.strptime(first[t], '%Y-%m-%dT%H:%M:%SZ') >= target)
    data = json.load(open(output_path, encoding='utf-8')) if os.path.exists(output_path) else {}
    stubs = set(json.load(open(_skip_path(output_path)))) if os.path.exists(_skip_path(output_path)) else set()
    todo = [t for t in post if _key(t) not in data and _key(t) not in stubs]
    print(f"C_post: {len(post)} of {len(cands)} articles created after the cutoff; {len(todo)} to fetch")

    def record(t, rec):
        with lock:
            if rec is None:
                stubs.add(_key(t))
            else:
                data[_key(t)] = rec

    failing += _run_passes(todo, lambda t: fetch_post_page(t, first[t]), record, workers, passes, 'C_post')
    _save_json(data, output_path)
    _save_json(sorted(stubs), _skip_path(output_path))
    print(f"C_post: {len(data)} pages -> {output_path} ({len(stubs)} redirects or stubs skipped)")
    return failing


# --- Filtering: pages that are not real, current base-game mechanics -------------------------------


STORY_TITLE_RE = re.compile(
    r"^(Minecraft:_|Minecraft_Dungeons|Minecraft_Legends|Minecraft_Earth|Minecraft_Story)",
    re.IGNORECASE,
)


CHAPTER_TITLE_RE = re.compile(r"/\d*CHP$|/[A-Za-z]+\d*CHP$")


EDITION_TITLE_RE = re.compile(
    r"^(Pocket_Edition|Education_Edition|Legacy_Console_Edition|New_Nintendo_3DS_Edition"
    r"|Minecraft_Education_\d|China_Edition|Pi_Edition|Console_Edition)"
    r"|^A_sneak_peek_at|^\d+_Years_of_Minecraft|^The_Evolution_of_Minecraft",
    re.IGNORECASE,
)


RESIDUAL_TITLE_RE = re.compile(
    r"\((?:Legacy_Console_Edition|Bedrock_Edition|Education_Edition|Pocket_Edition|Console_Edition)\)"
    r"|is_now_available|now_available_on|^Minecraft_is_"
    r"|/Plot$|/ED$|_Explorers$|^Mobs_of_Minecraft|^Encyclopedia$",
    re.IGNORECASE,
)


NON_ARTICLE_TITLE_RE = re.compile(
    r"\(disambiguation\)$|/Asset_history$|/Structure_history$|/Gallery$"
    r"|^Bedrock_Editor_|^Launcher_\d"
    r"|\.(json|png|mclevel)$|-atlas$"
    r"|_Cape($|!|_\()"
    r"|^Minecraft_Blast|\(Blast\)$"
    r"|^Calculators/|^Invalid_data_value_|^List_of_"
    r"|_version_history$|^Update_\d|^[A-Za-z]+_\d{4}_Update$",
    re.IGNORECASE,
)


MEDIA_CATEGORIES = {
    # print
    "Comic_books", "Book_characters", "Book_locations", "Books",
    "Minecraft_(Dark_Horse_Comics)_series",
    # film and series
    "A_Minecraft_Movie", "A_Minecraft_Movie_Squared", "Minecraft_Mini-Series",
    "Minecraft_Mini-Series_characters", "Minecraft_Mini-Series_locations", "Mini-Series_episodes",
    "M.A.R.I.L.L.A._and_Narrator_series", "Hat_Films", "Video_series",
    "Media_featuring_fanworks", "Media_using_Vibrant_Visuals",
    # merchandise and brand collaborations
    "Collaborations", "Collaboration_character_items", "Merchandise", "Food_products",
    "LEGO", "LEGO_sets",
    # events and promotion
    "Events", "Event_servers", "Promotional_characters", "Cross-franchise_promotions",
    "DLC_promotions",
    # paid add-ons
    "Skin_packs", "Collaborative_skin_packs", "Maps_with_skin_packs", "Mash-up_packs",
    "Texture_packs", "Minecraft_Marketplace",
    # other
    "Educational_maps", "Lost_media",
    # people
    "Community", "YouTubers", "Actors", "Musicians", "Writers", "Mojang_Studios", "MINECON",
    "MC_Championship", "Guest_characters", "Villains",
    # video and online media
    "Animated_content", "Live_action_content", "Music_videos", "Vocal_songs", "Game_trailers",
    "Comedies", "Element_Animation", "Minecraft:_Story_Mode",
    # other products, games and companies
    "Companies", "Software", "Websites", "Tabletop_games", "Mojam_games", "Non-Minecraft_games",
    "Adventure_games", "Science_fiction",
    # novels and fiction
    "Meta_novels", "Nick_Eliopulos_novels", "Fiction",
    # events and tools
    "TwitchCon", "Interactive_tools_and_calculators",
    # Deliberately absent, found by review: "Server" (tags Commands, Target_selectors,
    # server.properties), "Online_content" (tags Steve, Alex and the default
    # characters) and "Minecraft_(franchise)" (tags the Minecraft page itself).
}


OFFGAME_TITLES = {
    "GenWars", "Mob_Maze", "SoulSteel",
    "10th_Anniversary", "Ask_Mojang", "Biome_Bucket_List", "Minecraft_Mash-Up",
    "Minecraft_Support_Virtual_Agent", "Shape_Your_World:_Minecraft_meets_Art",
    "The_Togetherness_Machine", "Doki_Doki_Compatibility_Quest_III", "Home_Chef", "Dole",
    "Community_Challenge", "Minecraft_Creator-Safe_Playlist", "Java_Realms_Content_Creator_Program",
    "Java_account_migration", "Magazine", "Inbox", "Starfield", "Vibrant_Adventures",
    "Cyber_Defender", "Good_Game", "Learning_Cubed_Update", "Atlas.terrain",
    "Reed_Smart:_AI_Detective/An_Ode_to_Deception", "Reed_Smart:_AI_Detective/Dine_&_Deceive",
    "Reed_Smart:_AI_Detective/The_Deepest_Fake",
    "API_testing_capes", "Applet", "MD3_Test_Mob", "Historical_panoramas", "Removed_textures",
    "Texture_atlas/Predefined/Removed_from_Java_Edition/User_interface_atlases",
}


def _is_creator_doc(categories: List[str]) -> bool:
    return (any("Creator_Features" in c for c in categories)
            or "Data_packs" in categories
            or any("Add-on" in c for c in categories))


MEDIA_KEEP_TITLES = {
    "Emote", "Character_Creator", "Adventure_Update",
    "Steve", "Alex", "Ari", "Efe", "Kai", "Noor", "Sunny", "Zuri",
}


def _media_category(categories: List[str]) -> str:
    """Returns the first media/merchandise/marketing category, or ''."""
    return next((c for c in categories if c in MEDIA_CATEGORIES), "")


def title_reason(title: str) -> str:
    """Returns a rejection reason from the title alone, or '' to keep."""
    if STORY_TITLE_RE.match(title) or CHAPTER_TITLE_RE.search(title):
        return "story/spin-off"
    if is_blacklisted(f"https://minecraft.wiki/w/{title}"):
        return "version/changelog"
    if EDITION_TITLE_RE.search(title):
        return "other-edition/news"
    if RESIDUAL_TITLE_RE.search(title):
        return "encyclopedia/announcement"
    if NON_ARTICLE_TITLE_RE.search(title):
        return "non-article page"
    if title in OFFGAME_TITLES:
        return "off-game (reviewed)"
    return ""


def classify(item: Tuple[str, Dict[str, Any]], fetch_categories: bool) -> Tuple[str, str, List[str]]:
    """Returns (title, reason, categories). An empty reason means keep."""
    title, data = item
    reason = title_reason(title)
    if reason:
        return title, reason, data.get("categories", [])

    cats = data.get("categories")
    if cats is None and fetch_categories:
        cats = get_page_categories(title)
        if cats is None:
            # A failed lookup is not evidence of innocence; keeping the page
            # risks admitting joke content, so flag it for review.
            return title, "category lookup failed", []
    offending = is_excluded_category(cats or [])
    if offending:
        return title, f"joke/April Fools ({offending})", cats or []
    if _is_creator_doc(cats or []):
        return title, "creator/add-on docs", cats or []
    media = "" if title in MEDIA_KEEP_TITLES else _media_category(cats or [])
    if media:
        return title, f"media/merchandise ({media})", cats or []
    return title, "", (cats or [])


# Rules added for the 2026 rebuild (first author, 23 Oct 2026). The lists above predate many pages created since
# (Heelys, Batman (world), ...) and were tuned on a capped crawl that rarely reached sub-pages.
EXTRA_TITLE_RULES = {
    "structure blueprints": re.compile(r"/Blueprints(/|$)"),            # /Structure overviews stay
    "Character Creator cosmetics": re.compile(r"^Character_Creator(/|$)"),
    "dated / version-line pages": re.compile(r"^\d{8}$|^\d+\.x$"),
    "sound / outdated-loot / store sub-pages": re.compile(r"/Sounds$|/Outdated_loot_tables|_store_pages$"),
    "protocol / release-timeline / API pages": re.compile(r"Protocol_version_numbers|release_timeline|_API(/|$)"),
}
EXTRA_CATEGORY_RULES = {
    "disambiguation pages": lambda c: "Disambiguation_pages" in c,
    "soundtrack tracks": lambda c: any("Original_Game_Soundtrack" in x for x in c) and "Music_discs" not in c,
    "maps and collaborations": lambda c: bool({"Maps", "Collaborative_maps", "Other_franchises"} & c),
    "franchise and merchandise": lambda c: "Minecraft_(franchise)" in c,
    "novels and book settings": lambda c: any(x.endswith("_novels") for x in c) or bool({"Schools", "Earth_locations"} & c),
}
# C_post only. Set-index pages (Plant, Metal, ...) carry facts in C_pre. Tutorials created after the cutoff are guides
# to old mechanics (TNT cannons, minecart trains, the deep dark): 65 of the first 100 unanswerable drafts targeted
# them with pre-cutoff facts ("Tamed wolves can be brought into mineshafts").
POST_ONLY_TITLE_RULES = {"tutorials (guides to old mechanics)": re.compile(r"^Tutorial:")}
POST_ONLY_CATEGORY_RULES = {
    "set-index pages": lambda c: "Set_index_pages" in c,
    "planned, unreleased features": lambda c: "Planned" in c,
}
EXTRA_KEEP_TITLES = {"Minecraft"}
DISAMBIGUATION_MAX_CHARS = 1500


def extra_reason(title: str, categories: List[str], post: bool) -> str:
    """The 2026 rules: a rejection reason, or '' to keep."""
    if title in EXTRA_KEEP_TITLES:
        return ""
    for name, rx in list(EXTRA_TITLE_RULES.items()) + (list(POST_ONLY_TITLE_RULES.items()) if post else []):
        if rx.search(title):
            return name
    c = set(categories or [])
    for name, f in list(EXTRA_CATEGORY_RULES.items()) + (list(POST_ONLY_CATEGORY_RULES.items()) if post else []):
        if f(c):
            return name
    return ""


def page_reason(title: str, record: Dict[str, Any], current: List[str], cutoff: Optional[List[str]]) -> str:
    """Why a page is dropped ('' to keep). C_post pages (cutoff None) are judged on their current categories, which
    are those of the stored revision. C_pre pages are stored as they were at the cutoff, so:
      - topic rules (joke, media, creator docs, franchise, maps, novels, soundtrack) read the current categories:
        a page's topic does not change, and cutoff-time categories over-drop ("Block states" carried a joke
        category on 31 March 2025 from an April Fools section);
      - the page-type rule (disambiguation) reads the cutoff revision's categories, and only for text under 1,500
        characters (a link list): "Ocean Ruins" was a 14,000-character article at the cutoff and became a
        disambiguation page in April 2026, while summary pages tagged disambiguation then (Metal, Golem) carry facts."""
    if cutoff is None:
        return classify((title, {"categories": current}), False)[1] or extra_reason(title, current, post=True)
    topic = sorted(set(current) - {"Disambiguation_pages"})
    reason = classify((title, {"categories": topic}), False)[1] or extra_reason(title, topic, post=False)
    if not reason and "Disambiguation_pages" in cutoff and len(record.get("content") or "") < DISAMBIGUATION_MAX_CHARS:
        reason = "disambiguation page at the cutoff"
    return reason


def lookup_categories(title: str, attempts: int = 6) -> List[str]:
    """Current categories of a page not in the page list. Raises if every attempt fails: a failed lookup is
    neither evidence of innocence (joke pages would get in) nor of guilt (Phantom and Explosion were dropped)."""
    for i in range(attempts):
        cats = get_page_categories(title)
        if cats is not None:
            return cats
        time.sleep(10 * (i + 1))
    raise WikiAPIError(f"category lookup failed for {title}")


def cutoff_categories(oldid: str) -> List[str]:
    """Categories of a stored revision."""
    d = _api({"action": "parse", "oldid": oldid, "prop": "categories"})
    if "parse" not in d:
        raise WikiAPIError(f"no parse result for revision {oldid}")
    return [c["*"] for c in d["parse"].get("categories", [])]


def filter_corpus(path: str, label: str, dry_run: bool, workers: int, post: bool) -> List[str]:
    """Filters one corpus JSON; returns the titles it dropped (or would drop). The unfiltered corpus is kept as
    <path>.prefilter and is what a rerun filters, so changed rules apply to every page."""
    backup = f"{path}.prefilter"
    src = backup if os.path.exists(backup) else path
    if not os.path.exists(src):
        print(f"[{label}] missing: {path}")
        return []
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n[{label}] {len(data)} pages in {src}")

    listed = {}
    if os.path.exists(PAGE_LIST_FILE):
        listed = {_key(t): c for t, c in _load_page_list()["articles"].items()}
    # A C_pre page missing from the list (an article at the cutoff, a redirect now) is judged on its cutoff
    # categories; C_post pages, and every page when there is no list, are looked up.
    lookups = [t for t in data if t not in listed and not title_reason(t) and (post or not listed)]
    print(f"[{label}] categories: {len(data) - len(lookups)} from the page list, {len(lookups)} to look up")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        current = dict(zip(lookups, ex.map(lookup_categories, lookups)))
    current.update({t: listed[t] for t in data if t in listed})

    cut = None
    if not post:
        cache_path = f"{path}.cutoff_categories.json"
        cut = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
        todo = [t for t in data if t not in cut and not (title_reason(t) or extra_reason(t, [], post=False))]
        print(f"[{label}] cutoff-revision categories: {len(todo)} to fetch")
        lock = threading.Lock()

        def record(t, cats):
            with lock:
                cut[t] = cats

        failing = _run_passes(todo, lambda t: cutoff_categories(data[t]["oldid"]), record, workers, 4,
                              f"{label} categories")
        _save_json(cut, cache_path)
        if failing:
            raise WikiAPIError(f"cutoff categories still failing for {len(failing)} pages; rerun the filter")

    dropped: Dict[str, str] = {}
    kept: Dict[str, Any] = {}
    for title, rec in data.items():
        reason = (title_reason(title) or extra_reason(title, [], post)
                  or page_reason(title, rec, current.get(title, (cut or {}).get(title, [])),
                                 None if post else cut.get(title, [])))
        if reason:
            dropped[title] = reason
        else:
            kept[title] = {**rec, "categories": current[title] if post else cut[title]}

    by_reason: Dict[str, List[str]] = {}
    for t, r in dropped.items():
        by_reason.setdefault(r.split(" (")[0], []).append(t)
    print(f"[{label}] dropping {len(dropped)}, keeping {len(kept)}")
    for reason, titles in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"    {reason}: {len(titles)}")
        for t in titles[:8]:
            print(f"        {t}")
        if len(titles) > 8:
            print(f"        ... and {len(titles)-8} more")
    if dry_run:
        print(f"[{label}] dry run, nothing written")
        return list(dropped)

    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"[{label}] backup -> {backup}")
    _save_json(kept, path)
    _save_json(dropped, f"{path}.dropped.json")
    chars = sum(len(v.get("content") or "") for v in kept.values())
    print(f"[{label}] wrote {len(kept)} pages ({chars/1e6:.1f}M chars) -> {path}")
    return list(dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list", help="list every article (with categories) and recently edited redirect")
    for name, help_ in (("pre", "C_pre from the page list"), ("post", "C_post from the page list")):
        p = sp.add_parser(name, help=help_)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--passes", type=int, default=4, help="retry passes for pages whose requests failed")
    f = sp.add_parser("filter", help="filter both corpora in place (the unfiltered corpus is kept as .prefilter)")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("crawl-pre", help="the paper's link-following crawl of C_pre with a page cap")
    p.add_argument("--max-pages", type=int, default=4000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-categories", action="store_true", help="also seed from gameplay categories")
    p.add_argument("--output", default=TRAIN_JSON_FILE)
    q = sp.add_parser("crawl-post", help="the paper's link-following crawl of C_post with a page cap")
    q.add_argument("--max-pages", type=int, default=300)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--output", default=TEST_JSON_FILE)
    args = ap.parse_args()

    failing: List[str] = []
    if args.cmd == "list":
        list_pages()
    elif args.cmd == "pre":
        failing = scrape_pre_full(workers=args.workers, passes=args.passes)
    elif args.cmd == "post":
        failing = scrape_post_full(workers=args.workers, passes=args.passes)
    elif args.cmd == "filter":
        filter_corpus(TRAIN_JSON_FILE, "C_pre", args.dry_run, args.workers, post=False)
        filter_corpus(TEST_JSON_FILE, "C_post", args.dry_run, args.workers, post=True)
    elif args.cmd == "crawl-pre":
        scrape_wiki_pre_cutoff(start_urls=START_PAGES, output_path=args.output, max_pages=args.max_pages,
                               concurrency=args.workers, seed=args.seed,
                               seed_categories=SEED_CATEGORIES if args.seed_categories else None)
    else:
        scrape_wiki_post_cutoff(output_path=args.output, max_pages=args.max_pages, concurrency=args.workers)
    if failing:
        raise SystemExit(f"{len(failing)} pages still failing (e.g. {failing[:5]}); rerun to retry them")


if __name__ == "__main__":
    main()
