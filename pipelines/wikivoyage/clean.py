"""Convert Wikivoyage wikitext into markdown-like plain text.

Keeps the information that matters for retrieval:
- Section headings (== X ==) become markdown headings (## X) so the
  existing chunker can split on them.
- vCard/Marker templates (points of interest: hotels, sights, restaurants)
  are rendered as short text lines instead of being dropped.
- The IstIn breadcrumb template is extracted for the region hierarchy.
"""

import re

import mwparserfromhell

_ISTIN_RE = re.compile(r"\{\{\s*IstIn(?:Kat)?\s*\|([^}|]+)", re.IGNORECASE)
_H2_RE = re.compile(r"^==([^=].*?)==\s*$", re.MULTILINE)
_H3PLUS_RE = re.compile(r"^===+(.*?)===+\s*$", re.MULTILINE)

# vCard params worth keeping, in render order
_VCARD_KEEP = ["name", "type", "address", "directions", "phone", "url", "price", "description"]


def extract_parent(wikitext: str) -> str | None:
    """Return the IstIn parent region, if declared."""
    m = _ISTIN_RE.search(wikitext)
    return m.group(1).strip() if m else None


def _render_vcard(tpl) -> str:
    parts = []
    for key in _VCARD_KEEP:
        if tpl.has(key):
            value = str(tpl.get(key).value).strip()
            if value:
                parts.append(value if key in ("name", "description") else f"{key}: {value}")
    return "- " + ", ".join(parts) if parts else ""


def extract_pois(wikitext: str) -> list[dict]:
    """Extract named coordinates from vCard/Marker templates.

    Returns [{"name": ..., "lat": ..., "lon": ...}] for every template that
    carries usable coordinates.
    """
    pois: list[dict] = []
    code = mwparserfromhell.parse(wikitext)
    for tpl in code.filter_templates(recursive=True):
        tpl_name = str(tpl.name).strip().lower()
        if tpl_name not in ("vcard", "marker"):
            continue
        try:
            name = str(tpl.get("name").value).strip() if tpl.has("name") else ""
            lat_param = "lat" if tpl.has("lat") else None
            lon_param = "long" if tpl.has("long") else ("lon" if tpl.has("lon") else None)
            if not (name and lat_param and lon_param):
                continue
            lat = float(str(tpl.get(lat_param).value).strip())
            lon = float(str(tpl.get(lon_param).value).strip())
        except (ValueError, AttributeError):
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            pois.append({"name": name, "lat": lat, "lon": lon})
    return pois


def centroid(pois: list[dict]) -> dict | None:
    """Median coordinate of the article's POIs, robust against outliers."""
    if not pois:
        return None
    lats = sorted(p["lat"] for p in pois)
    lons = sorted(p["lon"] for p in pois)
    mid = len(pois) // 2
    return {"lat": lats[mid], "lon": lons[mid]}


_H2_SENTINEL = "XH2MARKERX"


def to_markdown(wikitext: str) -> str:
    """Wikitext -> markdown-ish plain text with ## section headings.

    A literal "## " cannot be inserted before parsing because a leading "#"
    is wikitext list syntax and strip_code would remove it. Headings are
    therefore tagged with a sentinel that survives strip_code and is
    converted to markdown afterwards.
    """
    text = _H3PLUS_RE.sub(lambda m: f"**{m.group(1).strip()}**", wikitext)
    text = _H2_RE.sub(lambda m: f"{_H2_SENTINEL}{m.group(1).strip()}", text)

    code = mwparserfromhell.parse(text)
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if name in ("vcard", "marker"):
            try:
                code.replace(tpl, _render_vcard(tpl))
            except ValueError:
                pass  # already removed as part of an outer node

    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.sub(rf"^\s*{_H2_SENTINEL}(.*)$", r"## \1", plain, flags=re.MULTILINE)
    # Leftover artifacts: file links and image caption fragments like "mini|..."
    plain = re.sub(r"^\s*(Datei|File|Bild):.*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(
        r"^\s*(mini|thumb|hochkant|links|rechts|zentriert|\d+px)\|.*$",
        "",
        plain,
        flags=re.MULTILINE,
    )
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip()
