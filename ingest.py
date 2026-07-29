"""
Ingestion script for the travel-agent project.

Given a COUNTRY name, this script:
  1. Fetches the country's Wikivoyage article
  2. Auto-discovers its cities from the "Cities" (and optionally
     "Other destinations") section
  3. Fetches Wikivoyage + Wikipedia articles for the country itself and
     every discovered city
  4. Strips wikitext markup and splits each article into section-level
     chunks
  5. Writes everything to a JSON file ready for embedding/indexing

Setup:
    pip install requests mwparserfromhell

Usage:
    python ingest.py --country Vietnam
    python ingest.py --country Vietnam --max-cities 6 --include-other-destinations
    python ingest.py --country Vietnam --out data/vietnam_chunks.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

try:
    import mwparserfromhell
    HAVE_MWPARSER = True
except ImportError:
    HAVE_MWPARSER = False

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia blocks requests with no/generic User-Agent (see their API etiquette
# policy) — set a real one identifying this script and a contact point.
HEADERS = {
    "User-Agent": "llm-zoomcamp-travel-agent/0.1 (student project; contact: your-email@example.com)"
}

# Section headers we don't want as standalone chunks (nav/boilerplate)
SKIP_SECTIONS = {"references", "external links", "see also", "connect"}


def fetch_wikitext(title: str, api_url: str) -> str | None:
    """Fetch raw wikitext for a page title from a MediaWiki API."""
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "wikitext",
        "redirects": 1,
    }
    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        print(f"  [!] {title}: {data['error'].get('info', 'not found')}")
        return None
    return data["parse"]["wikitext"]["*"]


def clean_wikitext(raw: str) -> str:
    """Strip markup down to plain readable text."""
    if HAVE_MWPARSER:
        wikicode = mwparserfromhell.parse(raw)
        text = wikicode.strip_code()
    else:
        # Fallback: regex-based cleanup if mwparserfromhell isn't installed
        text = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.DOTALL)  # templates
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # links
        text = re.sub(r"'''?", "", text)  # bold/italic
        text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)  # refs
        text = re.sub(r"<.*?>", "", text)  # stray html tags
    # Collapse excess blank lines/whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def split_into_sections(raw: str) -> list[dict]:
    """
    Split raw wikitext into (section_name, section_text) chunks using
    '== Heading ==' / '=== Subheading ===' boundaries.
    """
    # Capture heading level + name, keep everything up to the next heading
    pattern = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    matches = list(pattern.finditer(raw))

    sections = []
    if not matches or matches[0].start() > 0:
        intro_end = matches[0].start() if matches else len(raw)
        intro = raw[:intro_end].strip()
        if intro:
            sections.append(("Intro", intro))

    for i, m in enumerate(matches):
        name = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            sections.append((name, body))

    return sections


def split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Split `text` into pieces of at most `max_chars`, breaking on sentence
    boundaries where possible, with each piece repeating the trailing
    `overlap_chars` of the previous one so context isn't lost at the seam.
    If text already fits within max_chars, returns it as a single piece.
    """
    if len(text) <= max_chars:
        return [text]

    # Split into sentences (keeps the delimiter attached to the sentence)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        # current piece is full — flush it
        if current:
            pieces.append(current)
        # start the next piece with overlap carried over from the end of `current`
        overlap = current[-overlap_chars:] if current else ""
        current = f"{overlap} {sentence}".strip() if overlap else sentence

        # edge case: a single sentence longer than max_chars on its own —
        # hard-split it rather than looping forever
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars - overlap_chars:]

    if current:
        pieces.append(current)

    return pieces


# Wikivoyage country articles list cities under one of these section names
CITY_SECTION_NAMES = {"cities"}
OTHER_DEST_SECTION_NAMES = {"other destinations"}

# Namespace prefixes that are never real destination pages
BAD_PREFIXES = ("file:", "image:", "category:", "wikipedia:", "w:", "wikt:", "commons:")


def extract_links(raw_section_text: str) -> list[str]:
    """Pull [[Page]] / [[Page|Display]] wikilink targets out of raw wikitext."""
    links = re.findall(r"\[\[([^\|\]#]+)(?:#[^\|\]]*)?(?:\|[^\]]*)?\]\]", raw_section_text)
    seen, out = set(), []
    for link in links:
        title = link.strip().replace(" ", "_")
        if not title or title.lower().startswith(BAD_PREFIXES):
            continue
        if title not in seen:
            seen.add(title)
            out.append(title)
    return out


def discover_cities(country: str, max_cities: int, include_other: bool) -> list[str]:
    """Fetch the country's Wikivoyage page and pull out its city article titles."""
    raw = fetch_wikitext(country, WIKIVOYAGE_API)
    if not raw:
        raise SystemExit(f"Could not fetch Wikivoyage page for country: {country}")

    wanted_sections = set(CITY_SECTION_NAMES)
    if include_other:
        wanted_sections |= OTHER_DEST_SECTION_NAMES

    cities: list[str] = []
    for section_name, section_raw in split_into_sections(raw):
        if section_name.lower() in wanted_sections:
            for title in extract_links(section_raw):
                if title.lower() != country.lower() and title not in cities:
                    cities.append(title)

    if not cities:
        raise SystemExit(
            f"No 'Cities' section found for {country} — check the Wikivoyage "
            f"page title is correct (e.g. use underscores: 'United_States')."
        )

    return cities[:max_cities]


def build_chunks(
    city: str,
    source: str,
    title: str,
    raw_wikitext: str,
    max_chunk_chars: int = 800,
    overlap_chars: int = 150,
) -> list[dict]:
    chunks = []
    for section_name, section_raw in split_into_sections(raw_wikitext):
        if section_name.lower() in SKIP_SECTIONS:
            continue
        text = clean_wikitext(section_raw)
        if len(text) < 30:  # skip near-empty sections
            continue

        pieces = split_with_overlap(text, max_chunk_chars, overlap_chars)
        for i, piece in enumerate(pieces):
            # Only suffix a part number when a section actually got split
            part_suffix = f"_{i + 1}" if len(pieces) > 1 else ""
            section_label = f"{section_name} ({i + 1}/{len(pieces)})" if len(pieces) > 1 else section_name
            chunks.append(
                {
                    "id": f"{source}:{title}:{section_name}{part_suffix}".lower().replace(" ", "_"),
                    "city": city,
                    "source": source,
                    "page_title": title,
                    "section": section_label,
                    "text": piece,
                }
            )
    return chunks


def fetch_and_chunk(place: str, tag: str, max_chunk_chars: int, overlap_chars: int) -> list[dict]:
    """Fetch Wikivoyage + Wikipedia for one place (city or country) and chunk it."""
    chunks = []

    wv_text = fetch_wikitext(place, WIKIVOYAGE_API)
    if wv_text:
        chunks.extend(build_chunks(tag, "wikivoyage", place, wv_text, max_chunk_chars, overlap_chars))
    time.sleep(0.5)  # be polite to the API

    wp_text = fetch_wikitext(place, WIKIPEDIA_API)
    if wp_text:
        chunks.extend(build_chunks(tag, "wikipedia", place, wp_text, max_chunk_chars, overlap_chars))
    time.sleep(0.5)

    return chunks


def ingest(
    country: str,
    max_cities: int,
    include_other: bool,
    out_path: Path,
    max_chunk_chars: int,
    overlap_chars: int,
) -> None:
    print(f"Discovering cities for: {country}")
    cities = discover_cities(country, max_cities, include_other)
    print(f"Found {len(cities)} cities: {', '.join(cities)}\n")

    all_chunks = []

    # Country-level page too — this is where visa/currency/"get in" info lives
    print(f"Fetching country page: {country}")
    all_chunks.extend(fetch_and_chunk(country, country, max_chunk_chars, overlap_chars))

    for city in cities:
        print(f"Fetching: {city}")
        all_chunks.extend(fetch_and_chunk(city, city, max_chunk_chars, overlap_chars))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_chunks)} chunks to {out_path}")
    by_place = {}
    for c in all_chunks:
        by_place[c["city"]] = by_place.get(c["city"], 0) + 1
    for place, n in by_place.items():
        print(f"  {place}: {n} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover a country's cities on Wikivoyage and ingest travel data."
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Country page title, e.g. 'Vietnam' (use underscores for multi-word names).",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=9,
        help="Max number of cities to pull from the country's 'Cities' section (default: 9, "
        "matching Wikivoyage's own convention of listing up to 9 cities per country).",
    )
    parser.add_argument(
        "--include-other-destinations",
        action="store_true",
        help="Also pull links from the 'Other destinations' section (parks, islands, etc.).",
    )
    parser.add_argument(
        "--out",
        default="data/chunks.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=800,
        help="Max characters per chunk before a section gets sub-split (default: 800).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=150,
        help="Characters of trailing context repeated at the start of the next "
        "sub-chunk when a section is split (default: 150).",
    )
    args = parser.parse_args()
    ingest(
        args.country,
        args.max_cities,
        args.include_other_destinations,
        Path(args.out),
        args.max_chunk_chars,
        args.overlap_chars,
    )


if __name__ == "__main__":
    main()