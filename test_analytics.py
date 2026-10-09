"""
test_analytics.py — pytest unit tests for analytics.py.

These tests use small, hand-built synthetic frames (and the generated
dataset, if present) to lock down the cleaning, funnel, KPI, spend, and
cohort logic.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import analytics as A


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def employers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employer_id": [1, 2, 3],
            "employer_name": ["Alpha", "Bravo", "Charlie"],
            "employer_size": ["Small", "Mid-market", "Large"],
            "industry": ["Retail", "Technology", "Healthcare"],
        }
    )


@pytest.fixture
def members() -> pd.DataFrame:
    # 6 clean base rows spanning every funnel stage, + one-defect-each rows.
    base = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5, 6],
            "employer_id": pd.array([1, 1, 2, 2, 3, 3], dtype="Int64"),
            "enrollment_month": [
                "2025-01",
                "2025-01",
                "2025-02",
                "2025-02",
                "2025-03",
                "2025-03",
            ],
            "monthly_allowance": [400.0, 500.0, 600.0, 700.0, 450.0, 550.0],
            # 4 Active, 1 Selected Plan, 1 Browsed -> funnel is 6/5/4 (approx)
            "funnel_stage": [
                "Active",
                "Active",
                "Active",
                "Active",
                "Selected Plan",
                "Browsed",
            ],
            "metal_tier": ["Silver", "Gold", "Bronze", "Platinum", "Silver", ""],
            "monthly_premium": [350.0, 600.0, 500.0, 800.0, 420.0, np.nan],
            "status": [
                "Active",
                "Active",
                "Active",
                "Active",
                "Inactive",
                "Inactive",
            ],
        }
    )

    # Each defect row carries exactly ONE defect so cleaning drop-counts are
    # unambiguous: dup key, null employer, orphan FK, out-of-range allowance,
    # invalid funnel stage.
    defects = pd.DataFrame(
        {
            "member_id": [1, 7, 8, 9, 10],  # member_id 1 duplicates a key
            "employer_id": pd.array([1, pd.NA, 99, 2, 3], dtype="Int64"),
            "enrollment_month": [
                "2025-01",
                "2025-02",
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            # row idx 3 = out-of-range allowance (9999), others fine
            "monthly_allowance": [400.0, 500.0, 600.0, 9999.0, 550.0],
            "funnel_stage": [
                "Active",
                "Browsed",
                "Active",
                "Active",
                "WHO_KNOWS",  # invalid stage
            ],
            "metal_tier": ["Silver", "", "Gold", "Bronze", "Silver"],
            "monthly_premium": [350.0, np.nan, 600.0, 500.0, 420.0],
            "status": ["Active", "Inactive", "Active", "Active", "Active"],
        }
    )
    return pd.concat([base, defects], ignore_index=True)


@pytest.fixture
def monthly() -> pd.DataFrame:
    # Active base members 1..4 each with two months of spend.
    rows = []
    spend = {
        1: [(("2025-01"), 400.0, 200.0), (("2025-02"), 400.0, 350.0)],
        2: [(("2025-01"), 500.0, 250.0), (("2025-02"), 500.0, 500.0)],
        3: [(("2025-02"), 600.0, 300.0), (("2025-03"), 600.0, 500.0)],
        4: [(("2025-02"), 700.0, 350.0), (("2025-03"), 700.0, 700.0)],
    }
    for mid, recs in spend.items():
        for month, allow, used in recs:
            rows.append(
                {
                    "member_id": mid,
                    "month": month,
                    "monthly_allowance": allow,
                    "allowance_used": used,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_clean_removes_all_defects(members, employers):
    clean, report = A.clean_members(members, employers)
    # Only the 6 pristine base rows should survive.
    assert report["rows_out"] == 6
    assert report["dropped_duplicate_key"] == 1
    assert report["dropped_null"] >= 1
    assert report["dropped_bad_employer_fk"] >= 1
    assert report["dropped_out_of_range_allowance"] >= 1
    assert report["dropped_invalid_stage"] >= 1
    assert clean["funnel_stage"].isin(A.FUNNEL_STAGES).all()
    assert clean["employer_id"].isin(employers["employer_id"]).all()


def test_clean_report_retention_pct(members, employers):
    _, report = A.clean_members(members, employers)
    assert 0 < report["pct_retained"] <= 100


# --------------------------------------------------------------------------- #
# Funnel
# --------------------------------------------------------------------------- #
def test_funnel_counts_monotonic(members, employers):
    clean, _ = A.clean_members(members, employers)
    funnel = A.funnel_counts(clean)
    assert list(funnel["stage"]) == A.FUNNEL_STAGES
    counts = funnel["n_members"].tolist()
    # Non-increasing down the funnel.
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    # 6 enrolled; 5 browsed-or-beyond; 5 selected-or-beyond; 4 active.
    assert counts[0] == 6
    assert counts[3] == 4


def test_funnel_step_conversion_bounds(members, employers):
    clean, _ = A.clean_members(members, employers)
    funnel = A.funnel_counts(clean)
    assert funnel["step_conversion"].between(0, 1).all()
    assert funnel["pct_of_enrolled"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def test_kpi_summary_values(members, employers, monthly):
    clean, _ = A.clean_members(members, employers)
    snapshot = A.month_snapshot(monthly, "2025-02")
    k = A.kpi_summary(clean, snapshot)
    assert k["total_members"] == 6
    assert k["active_members"] == 4
    # Snapshot 2025-02 has members 1,2,3,4 using 350+500+300+350 = 1500
    # against offered allowance 400+500+600+700 = 2200.
    used = 350.0 + 500.0 + 300.0 + 350.0
    offered = 400.0 + 500.0 + 600.0 + 700.0
    assert k["allowance_utilization"] == pytest.approx(used / offered)
    assert k["employer_monthly_spend"] == pytest.approx(used)
    assert k["avg_monthly_allowance"] == pytest.approx(
        clean["monthly_allowance"].mean()
    )


def test_kpi_avg_premium_active_only(members, employers, monthly):
    clean, _ = A.clean_members(members, employers)
    snapshot = A.month_snapshot(monthly, "2025-02")
    k = A.kpi_summary(clean, snapshot)
    active_prem = clean.loc[clean["status"] == "Active", "monthly_premium"].dropna()
    assert k["avg_monthly_premium"] == pytest.approx(active_prem.mean())


# --------------------------------------------------------------------------- #
# Spend breakdowns
# --------------------------------------------------------------------------- #
def test_spend_by_employer_size(members, employers, monthly):
    snapshot = A.month_snapshot(monthly, "2025-02")
    g = A.spend_by_employer_size(snapshot, members, employers)
    assert "employer_size" in g.columns
    # used + unused == offered per row
    assert np.allclose(g["used"] + g["unused"], g["offered"])
    assert g["utilization"].between(0, 1.0001).all()


def test_spend_by_cohort(members, employers, monthly):
    snapshot = A.month_snapshot(monthly, "2025-02")
    g = A.spend_by_cohort(snapshot, members)
    assert "enrollment_month" in g.columns
    assert (g["offered"] >= g["used"]).all()


def test_active_premiums(members, employers):
    clean, _ = A.clean_members(members, employers)
    prem = A.active_premiums(clean)
    # 4 active base members all have premiums.
    assert len(prem) == 4
    assert prem.notna().all()


# --------------------------------------------------------------------------- #
# Cohort over time
# --------------------------------------------------------------------------- #
def test_utilization_by_cohort_month(members, employers, monthly):
    cohort = A.utilization_by_cohort_month(monthly, members)
    assert set(
        ["enrollment_month", "month", "utilization", "n_members"]
    ).issubset(cohort.columns)
    assert cohort["utilization"].between(0, 1.0001).all()
    # members 1 & 2 both enrolled 2025-01; in 2025-01 they used 200+250
    # against offered 400+500 -> utilization 450/900 = 0.5
    row = cohort[
        (cohort["enrollment_month"] == "2025-01")
        & (cohort["month"] == "2025-01")
    ].iloc[0]
    assert row["utilization"] == pytest.approx((200.0 + 250.0) / (400.0 + 500.0))


# --------------------------------------------------------------------------- #
# DQ helpers
# --------------------------------------------------------------------------- #
def test_dq_helpers(members, employers):
    assert A.test_unique(members, "member_id")["status"] == "FAIL"
    assert A.test_not_null(members, "employer_id")["status"] == "FAIL"
    assert (
        A.test_accepted_values(members, "funnel_stage", A.FUNNEL_STAGES)[
            "status"
        ]
        == "FAIL"
    )
    assert (
        A.test_referential_integrity(
            members, "employer_id", employers, "employer_id"
        )["status"]
        == "FAIL"
    )
    # A clean column passes.
    assert A.test_not_null(members, "member_id")["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Empty-input safety
# --------------------------------------------------------------------------- #
def test_empty_frame_is_safe():
    empty_members = pd.DataFrame(
        columns=[
            "member_id",
            "employer_id",
            "enrollment_month",
            "monthly_allowance",
            "funnel_stage",
            "metal_tier",
            "monthly_premium",
            "status",
        ]
    )
    empty_monthly = pd.DataFrame(
        columns=["member_id", "month", "monthly_allowance", "allowance_used"]
    )
    k = A.kpi_summary(empty_members, empty_monthly)
    assert k["total_members"] == 0
    assert k["allowance_utilization"] == 0.0
    assert A.utilization_by_cohort_month(empty_monthly, empty_members).empty


# --------------------------------------------------------------------------- #
# Integration with the generated dataset (if present)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(A.DEFAULT_MEMBERS_PATH),
    reason="run `python data_gen.py` first to generate the dataset",
)
def test_generated_dataset_is_well_formed():
    employers = A.load_employers(A.DEFAULT_EMPLOYERS_PATH)
    members = A.load_members(A.DEFAULT_MEMBERS_PATH)
    monthly = A.load_monthly(A.DEFAULT_MONTHLY_PATH)

    clean, report = A.clean_members(members, employers)
    # The generator makes clean data, so retention should be high.
    assert report["pct_retained"] == 100.0
    assert report["rows_out"] > 0

    funnel = A.funnel_counts(clean)
    counts = funnel["n_members"].tolist()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))

    snapshot = A.month_snapshot(monthly, A.AS_OF_MONTH)
    assert len(snapshot) > 0
    k = A.kpi_summary(clean, snapshot)
    assert 0 < k["allowance_utilization"] <= 1.0
    assert k["active_members"] == len(snapshot)

    cohort = A.utilization_by_cohort_month(monthly, members)
    assert cohort["utilization"].between(0, 1.0001).all()
