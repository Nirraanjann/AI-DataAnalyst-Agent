"""
tests/test_analyst_queries.py

Verifies each Phase 2 analyst_queries function against the ground-truth
values already recorded in evaluation_questions.json (Phase 1). Expected
values are loaded at collection time, not hardcoded a second time here,
so this file and the JSON can't silently drift apart.

These are integration tests -- they require a working Postgres connection
(same .env already used by scripts/load_data.py / generate_ground_truth.py).
"""

import json
from pathlib import Path

import pytest

from app.database.connection import get_engine
from app.tools import analyst_queries as aq

EVAL_PATH = Path(__file__).resolve().parent.parent / "evaluation_questions.json"


def _load_expected() -> dict:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return {q["id"]: q["expected_value"] for q in questions}


EXPECTED = _load_expected()


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def assert_matches_expected(actual, expected):
    """Recursively compares actual vs expected: pytest.approx for floats
    (currency values can drift slightly from NUMERIC/float round-tripping),
    exact equality for everything else (strings, ints, None, dict keys)."""
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-3)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key, exp_val in expected.items():
            assert_matches_expected(actual[key], exp_val)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_matches_expected(actual_item, expected_item)
    else:
        assert actual == expected


def test_get_monthly_revenue(engine):
    assert_matches_expected(aq.get_monthly_revenue(engine), EXPECTED["q01"])


def test_get_top_products(engine):
    assert_matches_expected(aq.get_top_products(engine, limit=20), EXPECTED["q02"])


def test_get_fastest_growing_regions(engine):
    assert_matches_expected(aq.get_fastest_growing_regions(engine), EXPECTED["q03"])


def test_get_top_customers(engine):
    assert_matches_expected(aq.get_top_customers(engine, limit=10), EXPECTED["q04"])


def test_get_declining_products(engine):
    assert_matches_expected(aq.get_declining_products(engine, limit=20), EXPECTED["q05"])


def test_get_average_order_value(engine):
    assert_matches_expected(aq.get_average_order_value(engine), EXPECTED["q06"])


def test_get_peak_revenue_month(engine):
    assert_matches_expected(aq.get_peak_revenue_month(engine), EXPECTED["q07"])


def test_get_avg_order_value_by_category(engine):
    assert_matches_expected(aq.get_avg_order_value_by_category(engine, limit=10), EXPECTED["q08"])


def test_get_revenue_anomalies(engine):
    assert_matches_expected(aq.get_revenue_anomalies(engine), EXPECTED["q09"])