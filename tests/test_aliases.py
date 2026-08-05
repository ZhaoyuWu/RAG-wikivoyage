"""Transliteration alias expansion for retrieval."""

from src.aliases import expand_query


def test_appends_german_spelling():
    assert "Düsseldorf" in expand_query("杜塞尔多夫老城")
    assert "Aachen" in expand_query("亚琛大教堂")
    assert "Lübeck" in expand_query("吕贝克老城")


def test_no_double_when_german_present():
    # German form already there -> don't append it again.
    out = expand_query("Aachen 亚琛大教堂")
    assert out.count("Aachen") == 1


def test_no_match_is_passthrough():
    q = "有什么好吃的餐厅"
    assert expand_query(q) == q


def test_longest_alias_wins():
    # 施特拉尔松德 must map to Stralsund, not match a shorter fragment.
    assert "Stralsund" in expand_query("施特拉尔松德海洋馆")


def test_region_alias():
    assert "Harz" in expand_query("哈茨徒步路线")
    assert "Schwarzwald" in expand_query("黑森林一日游")
