"""Chinese transliteration -> German place name aliases.

The embedding model and BM25 both fail on transliterated proper nouns:
"杜塞尔多夫" carries no signal that it means Düsseldorf, and the sparse leg
has no German token to match. The eval quantified this — it is the main
drag on the 56% golden-set hit rate.

Fix: when a query contains a known transliteration, append the German
spelling so BM25 gets an exact token to anchor on. This is deterministic
proper-noun knowledge, not something to leave to the model to guess.

Covers the major German travel cities/regions. Extend as gaps show up in
the eval — it is a lookup table, not an algorithm.
"""

import re

# Chinese transliteration -> canonical German spelling.
ALIASES: dict[str, str] = {
    # Big cities
    "杜塞尔多夫": "Düsseldorf",
    "科隆": "Köln",
    "慕尼黑": "München",
    "法兰克福": "Frankfurt",
    "汉堡": "Hamburg",
    "斯图加特": "Stuttgart",
    "杜伊斯堡": "Duisburg",
    "多特蒙德": "Dortmund",
    "埃森": "Essen",
    "不来梅": "Bremen",
    "德累斯顿": "Dresden",
    "莱比锡": "Leipzig",
    "纽伦堡": "Nürnberg",
    "汉诺威": "Hannover",
    "波恩": "Bonn",
    "亚琛": "Aachen",
    "明斯特": "Münster",
    "波茨坦": "Potsdam",
    "马格德堡": "Magdeburg",
    "威斯巴登": "Wiesbaden",
    "美因茨": "Mainz",
    "卡尔斯鲁厄": "Karlsruhe",
    "曼海姆": "Mannheim",
    "弗莱堡": "Freiburg",
    "海德堡": "Heidelberg",
    "维尔茨堡": "Würzburg",
    "奥格斯堡": "Augsburg",
    "雷根斯堡": "Regensburg",
    "特里尔": "Trier",
    # Hanseatic / north
    "吕贝克": "Lübeck",
    "罗斯托克": "Rostock",
    "施特拉尔松德": "Stralsund",
    "威廉港": "Wilhelmshaven",
    "基尔": "Kiel",
    "弗伦斯堡": "Flensburg",
    # Historic / tourist towns
    "罗滕堡": "Rothenburg",
    "班贝格": "Bamberg",
    "魏玛": "Weimar",
    "戈斯拉尔": "Goslar",
    "奎德林堡": "Quedlinburg",
    "维尔尼格罗德": "Wernigerode",
    "康斯坦茨": "Konstanz",
    # Regions
    "哈茨": "Harz",
    "黑森林": "Schwarzwald",
    "巴伐利亚": "Bayern",
    "莱茵河": "Rhein",
    "摩泽尔": "Mosel",
    "博登湖": "Bodensee",
    "萨克森": "Sachsen",
    "图林根": "Thüringen",
    "鲁尔区": "Ruhrgebiet",
    "阿尔卑斯": "Alpen",
    "北海": "Nordsee",
    "波罗的海": "Ostsee",
    # Capital (usually fine, but exact spelling helps the sparse leg)
    "柏林": "Berlin",
}

# Longest-first so "施特拉尔松德" matches before any shorter substring.
_SORTED = sorted(ALIASES, key=len, reverse=True)


def expand_query(query: str) -> str:
    """Append the German spelling of any transliteration found in the query,
    unless the German form is already present. Returns the query unchanged
    when nothing matches."""
    additions = []
    for zh in _SORTED:
        de = ALIASES[zh]
        if zh in query and not re.search(re.escape(de), query, re.IGNORECASE):
            additions.append(de)
    if not additions:
        return query
    return query + " " + " ".join(additions)
