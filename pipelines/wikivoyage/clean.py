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
