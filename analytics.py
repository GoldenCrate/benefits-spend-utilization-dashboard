"""
analytics.py — Framework-free metrics layer for ICHRA spend & utilization.

All functions here are pure pandas (no Streamlit import), so they are
unit-testable and reusable. Think of this module as the SQL/dbt metrics layer:
a cleaned staging pass over the raw feed, then a set of marts (funnel, KPIs,
spend, utilization, cohorts) that the Streamlit dashboard simply surfaces.

ALL INPUT DATA IS SYNTHETIC (see data_gen.py). Nothing here describes any
real person, employer, insurer, or health/claims record.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

# Reference / enum values (mirror dbt accepted_values tests).
FUNNEL_STAGES = ["Enrolled", "Browsed", "Selected Plan", "Active"]
METAL_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
EMPLOYER_SIZES = ["Small", "Mid-market", "Large"]
STATUS_ACTIVE = "Active"

# Plausible ranges for staging-layer range checks (USD / month).
ALLOWANCE_MIN = 50.0
ALLOWANCE_MAX = 2000.0

# Ordinal index of each funnel stage (for "furthest stage reached" logic).
_STAGE_INDEX = {s: i for i, s in enumerate(FUNNEL_STAGES)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_employers(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_members(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"metal_tier": "string"})
    df["employer_id"] = pd.to_numeric(
        df["employer_id"], errors="coerce"
    ).astype("Int64")
    df["monthly_allowance"] = pd.to_numeric(
        df["monthly_allowance"], errors="coerce"
    )
    df["monthly_premium"] = pd.to_numeric(
        df["monthly_premium"], errors="coerce"
    )
    df["metal_tier"] = df["metal_tier"].fillna("").astype(str)
    return df


def load_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["monthly_allowance"] = pd.to_numeric(
        df["monthly_allowance"], errors="coerce"
    )
    df["allowance_used"] = pd.to_numeric(df["allowance_used"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Staging / cleaning (dbt stg_* layer). Returns analysis frame + a report.
# --------------------------------------------------------------------------- #
def clean_members(
    members: pd.DataFrame, employers: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Return an analysis-ready member frame plus a report of what was removed.

    Cleaning rules (these mirror the kinds of dbt source tests a metrics layer
    would enforce before anything reaches a dashboard):
      - drop duplicate member_id keys (keep first)
      - drop rows with null employer_id or null allowance
      - drop rows whose employer_id is not a valid employer (referential)
      - drop rows with out-of-range monthly allowance
      - drop rows with an invalid funnel stage
    """
    report: dict = {"rows_in": len(members)}
    valid_emp_ids = set(employers["employer_id"].tolist())

    df = members.copy()

    before = len(df)
    df = df.drop_duplicates(subset="member_id", keep="first")
    report["dropped_duplicate_key"] = before - len(df)

    before = len(df)
    df = df[df["employer_id"].notna() & df["monthly_allowance"].notna()]
    report["dropped_null"] = before - len(df)

    before = len(df)
    df = df[df["employer_id"].isin(valid_emp_ids)]
    report["dropped_bad_employer_fk"] = before - len(df)

    before = len(df)
    df = df[
        (df["monthly_allowance"] >= ALLOWANCE_MIN)
        & (df["monthly_allowance"] <= ALLOWANCE_MAX)
    ]
    report["dropped_out_of_range_allowance"] = before - len(df)

    before = len(df)
    df = df[df["funnel_stage"].isin(FUNNEL_STAGES)]
    report["dropped_invalid_stage"] = before - len(df)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    report["pct_retained"] = (
        round(100.0 * len(df) / report["rows_in"], 1)
        if report["rows_in"]
        else 0.0
    )
    return df, report


# --------------------------------------------------------------------------- #
# Snapshot helper: one reporting month of the monthly spend fact
# --------------------------------------------------------------------------- #
def month_snapshot(monthly: pd.DataFrame, month: str) -> pd.DataFrame:
    """Rows of the monthly fact for a single reporting month."""
    return monthly[monthly["month"] == month].copy().reset_index(drop=True)


def latest_month(monthly: pd.DataFrame) -> str:
    if monthly.empty:
        return ""
    return str(monthly["month"].max())


# --------------------------------------------------------------------------- #
# Mart: plan-selection funnel
# --------------------------------------------------------------------------- #
def funnel_counts(clean: pd.DataFrame) -> pd.DataFrame:
    """Members reaching each funnel stage (furthest-stage-reached logic).

    A member counts toward every stage up to and including the furthest stage
    they reached, so the funnel is monotonically non-increasing.
    """
    stage_idx = clean["funnel_stage"].map(_STAGE_INDEX)
    rows = []
    top = len(clean)
    prev = None
    for i, stage in enumerate(FUNNEL_STAGES):
        n = int((stage_idx >= i).sum())
        step = 1.0 if prev is None else (n / prev if prev else 0.0)
        rows.append(
            {
                "stage": stage,
                "n_members": n,
                "pct_of_enrolled": (n / top if top else 0.0),
                "step_conversion": step,
            }
        )
        prev = n
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mart: KPI summary
# --------------------------------------------------------------------------- #
def kpi_summary(clean: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    """Headline KPIs. `snapshot` is one reporting month of the monthly fact."""
    total = len(clean)
    active = int((clean["status"] == STATUS_ACTIVE).sum())
    avg_allowance = (
        float(clean["monthly_allowance"].mean()) if total else 0.0
    )
    active_premiums = clean.loc[
        clean["status"] == STATUS_ACTIVE, "monthly_premium"
    ].dropna()
    avg_premium = float(active_premiums.mean()) if len(active_premiums) else 0.0

    used = float(snapshot["allowance_used"].sum()) if len(snapshot) else 0.0
    avail = float(snapshot["monthly_allowance"].sum()) if len(snapshot) else 0.0
    utilization = (used / avail) if avail else 0.0

    return {
        "total_members": total,
        "active_members": active,
        "avg_monthly_allowance": avg_allowance,
        "avg_monthly_premium": avg_premium,
        "allowance_utilization": utilization,
        # Employer's actual monthly outlay = allowance dollars reimbursed.
        "employer_monthly_spend": used,
        "allowance_offered": avail,
    }


# --------------------------------------------------------------------------- #
# Mart: spend & utilization breakdowns (single reporting month)
# --------------------------------------------------------------------------- #
def _spend_breakdown(
    snapshot: pd.DataFrame, keys: pd.DataFrame, dim: str
) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(
            columns=[dim, "n_members", "used", "unused", "offered", "utilization"]
        )
    s = snapshot.merge(keys, on="member_id", how="inner")
    g = (
        s.groupby(dim)
        .agg(
            n_members=("member_id", "nunique"),
            used=("allowance_used", "sum"),
            offered=("monthly_allowance", "sum"),
        )
        .reset_index()
    )
    g["unused"] = g["offered"] - g["used"]
    g["utilization"] = g["used"] / g["offered"]
    return g


def spend_by_employer_size(
    snapshot: pd.DataFrame, members: pd.DataFrame, employers: pd.DataFrame
) -> pd.DataFrame:
    keys = members[["member_id", "employer_id"]].merge(
        employers[["employer_id", "employer_size"]], on="employer_id", how="left"
    )[["member_id", "employer_size"]]
    g = _spend_breakdown(snapshot, keys, "employer_size")
    if g.empty:
        return g
    order = {s: i for i, s in enumerate(EMPLOYER_SIZES)}
    g["_o"] = g["employer_size"].map(order).fillna(99)
    return g.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def spend_by_cohort(
    snapshot: pd.DataFrame, members: pd.DataFrame
) -> pd.DataFrame:
    keys = members[["member_id", "enrollment_month"]]
    g = _spend_breakdown(snapshot, keys, "enrollment_month")
    if g.empty:
        return g
    return g.sort_values("enrollment_month").reset_index(drop=True)


def active_premiums(clean: pd.DataFrame) -> pd.Series:
    """Monthly premiums of active members (for the premium distribution)."""
    return clean.loc[
        clean["status"] == STATUS_ACTIVE, "monthly_premium"
    ].dropna()


# --------------------------------------------------------------------------- #
# Mart: cohort utilization over time
# --------------------------------------------------------------------------- #
def utilization_by_cohort_month(
    monthly: pd.DataFrame, members: pd.DataFrame
) -> pd.DataFrame:
    """Allowance utilization for each enrollment cohort, by calendar month.

    Returns a tidy frame: one row per (enrollment_month, month) with the
    dollars used, dollars offered, utilization, and member count. Early-tenure
    months show lower utilization (ramp-up), so recent cohorts sit lower.
    """
    if monthly.empty:
        return pd.DataFrame(
            columns=[
                "enrollment_month",
                "month",
                "n_members",
                "used",
                "offered",
                "utilization",
            ]
        )
    m = monthly.merge(
        members[["member_id", "enrollment_month"]], on="member_id", how="inner"
    )
    g = (
        m.groupby(["enrollment_month", "month"])
        .agg(
            n_members=("member_id", "nunique"),
            used=("allowance_used", "sum"),
            offered=("monthly_allowance", "sum"),
        )
        .reset_index()
    )
    g["utilization"] = g["used"] / g["offered"]
    return g.sort_values(["enrollment_month", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Data-quality helpers (dbt-style single-column tests), reused by tests
# --------------------------------------------------------------------------- #
def _result(test: str, column: str, n_failing: int, n_total: int) -> dict:
    return {
        "test": test,
        "column": column,
        "n_failing": int(n_failing),
        "n_total": int(n_total),
        "status": "PASS" if n_failing == 0 else "FAIL",
    }


def test_not_null(df: pd.DataFrame, column: str) -> dict:
    return _result("not_null", column, int(df[column].isna().sum()), len(df))


def test_unique(df: pd.DataFrame, column: str) -> dict:
    n_fail = int(df[column].duplicated(keep=False).sum())
    return _result("unique", column, n_fail, len(df))


def test_accepted_values(
    df: pd.DataFrame, column: str, allowed: Iterable
) -> dict:
    allowed = set(allowed)
    non_null = df[column].dropna()
    n_fail = int((~non_null.isin(allowed)).sum())
    return _result("accepted_values", column, n_fail, len(df))


def test_referential_integrity(
    df: pd.DataFrame, column: str, ref_df: pd.DataFrame, ref_column: str
) -> dict:
    valid = set(ref_df[ref_column].tolist())
    non_null = df[column].dropna()
    n_fail = int((~non_null.isin(valid)).sum())
    return _result("referential_integrity", column, n_fail, len(df))


# Convenience default paths ------------------------------------------------- #
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_EMPLOYERS_PATH = os.path.join(_DATA_DIR, "dim_employers.csv")
DEFAULT_MEMBERS_PATH = os.path.join(_DATA_DIR, "dim_members.csv")
DEFAULT_MONTHLY_PATH = os.path.join(_DATA_DIR, "fct_monthly_spend.csv")
# Mirrors AS_OF_MONTH in data_gen.py; used for the reporting-month snapshot.
AS_OF_MONTH = "2025-09"
