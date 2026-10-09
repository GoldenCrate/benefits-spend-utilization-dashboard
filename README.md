# Health-Benefits Spend & Utilization Analytics

A multi-view Streamlit dashboard for an **ICHRA** health-benefits platform —
the model where an employer gives each employee a fixed monthly **allowance**
to buy their own individual health plan, and reimburses premiums up to that
allowance. Built as a portfolio piece for a **Data Scientist, Analytics**
role, it is structured like a **dbt-modeled metrics layer surfaced in a BI
dashboard**: raw tables → a cleaned staging pass → funnel / KPI / spend /
cohort marts → charts.

> ## ⚠️ SYNTHETIC DATA — PLEASE READ
> **Every number in this project is 100% synthetic.** It is produced by a
> seeded random-number generator in [`data_gen.py`](./data_gen.py) and does
> **not** represent any real person, employer, insurer, health plan, or claims
> record. It contains **no protected health information** — none of the data
> describes a real individual. Employer names, allowances, premiums, and plan
> tiers are invented labels and numbers. Nothing here should be read as a
> real-world fact about health benefits, insurance, or spending.

---

## What ICHRA is (context)

ICHRA = *Individual Coverage Health Reimbursement Arrangement*. Instead of
offering a group plan, an employer gives each employee a tax-advantaged
monthly **allowance**. The employee **browses** individual plans, **selects**
one, and **activates** coverage; the employer reimburses the premium up to the
allowance. Allowance offered but not spent is generally **forfeited**, so
**allowance utilization** (dollars reimbursed ÷ dollars offered) is a core
business and member-value metric.

## Views

1. **Overview (KPIs)** — headline tiles: members enrolled, average monthly
   allowance, allowance utilization %, average monthly premium, and employer
   monthly spend, plus an offered-vs-used bar.
2. **Plan-selection funnel** — Enrolled → Browsed → Selected Plan → Active,
   with per-step conversion and the biggest drop-off highlighted.
3. **Spend & utilization** — allowance used vs. unused by **employer size**,
   and the **premium distribution** of selected plans against the average
   allowance (where the unused-dollar leakage comes from).
4. **Cohort utilization** — enrollment-month cohorts' allowance utilization
   over calendar time, showing the onboarding ramp to steady state.

Each view ends with a **"So what / recommendation"** callout.

## Data model (synthetic, dbt-shaped)

| Table | Grain | Purpose |
|-------|-------|---------|
| `dim_employers.csv` | one row per employer | size, industry lookup |
| `dim_members.csv` | one row per enrolled member | allowance, funnel stage, selected plan tier, premium, status |
| `fct_monthly_spend.csv` | one row per active member × month | allowance offered and allowance used that month |

`analytics.py` is the metrics layer: a staging-style `clean_members` pass
(dedupe keys, enforce not-null, referential integrity to employers, accepted
values for funnel stage, allowance range) followed by pure functions for the
funnel, KPIs, spend breakdowns, and cohort utilization. No Streamlit import
lives in that module, so every metric is unit-tested in isolation.

## Charts: Storytelling with Data

Visuals follow Cole Nussbaumer Knaflic's principles: **takeaway titles** (the
title states the insight), decluttered axes and no chartjunk, **direct value
labels** instead of gridlines, and a **restrained palette** (gray for context,
a single accent for emphasis; muted green/gray only for used/unused dollars).

## Project structure

| File | Purpose |
|------|---------|
| `data_gen.py` | Generates the synthetic warehouse (fixed seed). |
| `analytics.py` | Framework-free metrics layer: cleaning, funnel, KPIs, spend, cohorts. |
| `app.py` | Streamlit multi-view dashboard. |
| `test_analytics.py` | pytest unit tests for the metrics logic. |
| `requirements.txt` | Dependencies. |

## Run it locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (writes ./data/*.csv)
python data_gen.py

# 3. Run the tests (optional but recommended)
pytest -q

# 4. Launch the dashboard
streamlit run app.py
```

The `data/` folder is git-ignored because it is fully reproducible — just run
`python data_gen.py`.

## Tests

`test_analytics.py` covers the cleaning/row-retention logic (one defect per
row, so drop-counts are exact), the funnel monotonicity and step conversion,
the KPI math, the spend breakdowns (used + unused = offered), cohort
utilization, dbt-style data-quality helpers, and empty-input safety. Run with
`pytest -q`.

---

**Author:** Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)

*Portfolio demonstration on synthetic data. Built to show benefits-analytics
modeling and dashboarding; not affiliated with any organization and not based
on any real data or protected health information.*
