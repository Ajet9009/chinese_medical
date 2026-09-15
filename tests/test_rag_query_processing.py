from common.query_expansion import expand_zhongyi_query_step_01
from common.text_normalization import normalize_text_step_01, query_variants_step_02


def test_query_variants_include_simplified_and_traditional():
    variants = query_variants_step_02("人参味如何、主补什么？")

    assert "人参味如何、主补什么？" in variants
    assert any("人參" in item or "主補" in item for item in variants)


def test_normalize_text_converts_common_medical_terms():
    assert normalize_text_step_01("人參主補五臟", "simplified") == "人参主补五脏"
    assert normalize_text_step_01("麻黄汤治什么", "traditional") == "麻黃湯治什麼"


def test_expand_zhongyi_query_adds_aliases_without_llm():
    variants = expand_zhongyi_query_step_01("麻黄汤治什么？")

    assert any("麻黃湯" in item for item in variants)
    assert any("主治" in item or "功效" in item for item in variants)
