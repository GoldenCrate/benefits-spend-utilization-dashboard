"""
app.py — Health-Benefits Spend & Utilization Analytics (SYNTHETIC DATA).

A multi-view Streamlit dashboard built as a portfolio piece for a Data
Scientist, Analytics role on a health-benefits (ICHRA) platform. It surfaces a
dbt-style metrics layer (see analytics.py) across four views:

  1. Overview (KPIs)        — members, allowance, utilization, premium, spend
  2. Plan-selection funnel  — Enrolled -> Browsed -> Selected Plan -> Active
  3. Spend & utilization    — allowance used vs. unused by employer size/cohort
  4. Cohort utilization     — enrollment-month cohorts ramping over time

=========================== SYNTHETIC DATA NOTICE ===========================
Every number in this app is produced by a seeded random-number generator
(see data_gen.py). It does NOT represent any real person, employer, insurer,
health plan, or claims record, and contains no protected health information.
=============================================================================

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data": takeaway
titles, decluttered axes, direct labels, and restrained, purposeful color.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A

# --------------------------------------------------------------------------- #
# Page config + palette
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Health-Benefits Spend & Utilization (Synthetic)",
    page_icon="📊",
    layout="wide",
)

# Restrained palette: one neutral gray for context, one accent for emphasis.
GRAY = "#B8BEC7"
GRAY_DARK = "#6B7280"
ACCENT = "#1f6feb"  # emphasis blue
GOOD = "#2E7D5B"  # muted green (used / reimbursed)
BAD = "#B4452E"  # muted red (unused / forfeited)
INK = "#222529"


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load():
    employers = A.load_employers(A.DEFAULT_EMPLOYERS_PATH)
    members = A.load_members(A.DEFAULT_MEMBERS_PATH)
    monthly = A.load_monthly(A.DEFAULT_MONTHLY_PATH)
    return employers, members, monthly


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _recommendation(text: str) -> None:
    st.markdown(
        f"""
        <div style="background:#eef4ff;border-left:4px solid {ACCENT};
        padding:0.75rem 1rem;border-radius:4px;margin-top:0.5rem;">
        <strong>So what / recommendation.</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _synthetic_caption() -> None:
    st.caption(
        "⚠️ All figures are SYNTHETIC (seeded simulation) — not real people, "
        "employers, or claims. Built as an analytics portfolio demo."
    )


# --------------------------------------------------------------------------- #
# Chart helpers (Storytelling-with-Data styling)
# --------------------------------------------------------------------------- #
def _bare_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=INK)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=13),
        margin=dict(l=10, r=20, t=55, b=10),
        height=height,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRAY)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    d = funnel.copy()
    # Highlight the biggest single-step drop in the accent color.
    drop = 1.0 - d["step_conversion"]
    drop.iloc[0] = 0.0  # top stage has no prior step
    worst = int(drop.idxmax())
    colors = [GRAY if i != worst else ACCENT for i in range(len(d))]
    labels = [
        f"{n:,}  ({_fmt_pct(p)})"
        for n, p in zip(d["n_members"], d["pct_of_enrolled"])
    ]
    fig = go.Figure(
        go.Bar(
            x=d["n_members"],
            y=d["stage"],
            orientation="h",
            marker=dict(color=colors),
            text=labels,
            textposition="outside",
            textfont=dict(size=12, color=INK),
            cliponaxis=False,
        )
    )
    # Keep stages top-to-bottom in funnel order.
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["stage"][::-1]))
    fig = _bare_layout(
        fig,
        "Most drop-off happens at a single step of the plan-selection funnel",
        height=320,
    )
    fig.update_xaxes(visible=False, range=[0, d["n_members"].max() * 1.25])
    return fig


def stacked_used_unused(
    df: pd.DataFrame, label_col: str, title: str
) -> go.Figure:
    d = df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d[label_col].astype(str),
            y=d["used"],
            name="Allowance used (reimbursed)",
            marker=dict(color=GOOD),
        )
    )
    fig.add_trace(
        go.Bar(
            x=d[label_col].astype(str),
            y=d["unused"],
            name="Allowance unused (forfeited)",
            marker=dict(color=GRAY),
        )
    )
    # Direct-label utilization % above each stacked bar.
    fig.add_trace(
        go.Scatter(
            x=d[label_col].astype(str),
            y=d["offered"],
            mode="text",
            text=[_fmt_pct(v) for v in d["utilization"]],
            textposition="top center",
            textfont=dict(size=12, color=INK),
        )
    )
    fig = _bare_layout(fig, title, height=400)
    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(visible=True, tickprefix="$", showticklabels=True,
                     range=[0, d["offered"].max() * 1.18])
    return fig


def premium_histogram(premiums: pd.Series, allowance: float) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=premiums,
            marker=dict(color=GRAY, line=dict(color="white", width=1)),
            nbinsx=30,
        )
    )
    fig.add_vline(
        x=allowance,
        line=dict(color=ACCENT, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=allowance,
        yref="paper",
        y=1.0,
        text=f"Avg allowance {_fmt_usd(allowance)}",
        showarrow=False,
        font=dict(color=ACCENT, size=12),
        xanchor="left",
    )
    fig = _bare_layout(
        fig,
        "Many selected premiums sit below the allowance — so dollars go unused",
        height=360,
    )
    fig.update_xaxes(visible=True, tickprefix="$", showticklabels=True)
    fig.update_yaxes(visible=True, title="Members", showticklabels=True)
    return fig


def cohort_lines(cohort: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cohorts = sorted(cohort["enrollment_month"].unique())
    last = cohorts[-1] if cohorts else None
    for c in cohorts:
        d = cohort[cohort["enrollment_month"] == c].sort_values("month")
        is_last = c == last
        fig.add_trace(
            go.Scatter(
                x=d["month"],
                y=d["utilization"],
                mode="lines+markers",
                name=c,
                line=dict(
                    color=ACCENT if is_last else GRAY,
                    width=3 if is_last else 1.5,
                ),
                marker=dict(size=5),
            )
        )
        # Label each cohort line at its right end.
        tail = d.iloc[-1]
        fig.add_annotation(
            x=tail["month"],
            y=tail["utilization"],
            text=c,
            showarrow=False,
            xanchor="left",
            font=dict(
                size=11,
                color=ACCENT if is_last else GRAY_DARK,
            ),
        )
    fig = _bare_layout(
        fig,
        "Each cohort ramps to steady-state utilization over its first ~3 months",
        height=420,
    )
    fig.update_xaxes(visible=True, showticklabels=True)
    fig.update_yaxes(
        visible=True, showticklabels=True, tickformat=".0%", range=[0, 1.05]
    )
    fig.update_layout(margin=dict(r=70))
    return fig


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def view_overview(
    clean: pd.DataFrame, snapshot: pd.DataFrame, report: dict
) -> None:
    st.subheader("Benefits overview")
    _synthetic_caption()

    k = A.kpi_summary(clean, snapshot)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Members enrolled", f"{k['total_members']:,}")
    c2.metric("Avg monthly allowance", _fmt_usd(k["avg_monthly_allowance"]))
    c3.metric("Allowance utilization", _fmt_pct(k["allowance_utilization"]))
    c4.metric("Avg monthly premium", _fmt_usd(k["avg_monthly_premium"]))
    c5.metric("Employer spend / mo", _fmt_usd(k["employer_monthly_spend"]))

    st.markdown(
        f"Across **{k['total_members']:,}** enrolled members, "
        f"**{k['active_members']:,}** have active coverage. In the reporting "
        f"month ({A.AS_OF_MONTH}), employers offered "
        f"**{_fmt_usd(k['allowance_offered'])}** in allowance and reimbursed "
        f"**{_fmt_usd(k['employer_monthly_spend'])}** of it — an overall "
        f"allowance utilization of **{_fmt_pct(k['allowance_utilization'])}**."
    )

    # Offered vs. used (reporting month).
    offered = k["allowance_offered"]
    used = k["employer_monthly_spend"]
    fig = go.Figure(
        go.Bar(
            x=[used, offered - used],
            y=["Allowance (one month)", "Allowance (one month)"],
            orientation="h",
            marker=dict(color=[GOOD, GRAY]),
            text=[
                f"Used {_fmt_usd(used)}",
                f"Unused {_fmt_usd(offered - used)}",
            ],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=13, color="white"),
        )
    )
    fig = _bare_layout(
        fig,
        f"About {_fmt_pct(k['allowance_utilization'])} of offered allowance is "
        "actually reimbursed",
        height=200,
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch")

    _recommendation(
        "Utilization below 100% means employers fund an allowance that members "
        "do not fully spend (unused ICHRA dollars are typically forfeited). "
        "That is a retention and value story: help members pick plans that use "
        "more of their allowance, and the benefit feels more generous at no "
        "extra employer cost. The funnel and cohort views isolate where the "
        "leakage comes from."
    )


def view_funnel(clean: pd.DataFrame) -> None:
    st.subheader("Plan-selection funnel")
    _synthetic_caption()
    st.markdown(
        "Every enrolled member must **browse** plans, **select** one, and "
        "**activate** coverage before a single allowance dollar is used. Each "
        "step is a drop-off point."
    )

    funnel = A.funnel_counts(clean)
    st.plotly_chart(funnel_chart(funnel), width="stretch")

    with st.expander("Funnel detail (step conversion)"):
        show = funnel.copy()
        show["pct_of_enrolled"] = show["pct_of_enrolled"].map(_fmt_pct)
        show["step_conversion"] = show["step_conversion"].map(_fmt_pct)
        show.columns = [
            "Stage",
            "Members",
            "% of enrolled",
            "Step conversion",
        ]
        st.dataframe(show, width="stretch", hide_index=True)

    # Identify the weakest step for the recommendation.
    steps = funnel.iloc[1:].copy()
    steps["drop"] = 1.0 - steps["step_conversion"]
    worst = steps.loc[steps["drop"].idxmax()]
    _recommendation(
        f"The biggest leak is **{worst['stage']}**, where only "
        f"{_fmt_pct(worst['step_conversion'])} of the prior stage converts. "
        "Target that step first: if it is Browsed → Selected Plan, a guided "
        "plan-recommendation flow likely moves the metric more than top-of-"
        "funnel enrollment pushes do."
    )


def view_spend(
    snapshot: pd.DataFrame,
    members: pd.DataFrame,
    employers: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    st.subheader("Spend & utilization")
    _synthetic_caption()
    st.markdown(
        "Allowance **used** (green) is what the employer reimburses; "
        "**unused** (gray) is offered but forfeited. The percentage above each "
        "bar is utilization for that segment, in the reporting month."
    )

    tab_size, tab_premium = st.tabs(
        ["By employer size", "Premium vs. allowance"]
    )

    with tab_size:
        by_size = A.spend_by_employer_size(snapshot, members, employers)
        st.plotly_chart(
            stacked_used_unused(
                by_size,
                "employer_size",
                "Larger employers offer more allowance but leave more unused",
            ),
            width="stretch",
        )
        if not by_size.empty:
            best = by_size.loc[by_size["utilization"].idxmax()]
            worst = by_size.loc[by_size["utilization"].idxmin()]
            _recommendation(
                f"**{worst['employer_size']}** employers have the lowest "
                f"utilization ({_fmt_pct(worst['utilization'])}) vs. "
                f"**{best['employer_size']}** ({_fmt_pct(best['utilization'])}). "
                "Richer allowances outrun what members spend, so the marginal "
                "allowance dollar at large employers buys little extra "
                "coverage — worth flagging in employer QBRs."
            )

    with tab_premium:
        premiums = A.active_premiums(clean)
        avg_allow = (
            float(clean["monthly_allowance"].mean()) if len(clean) else 0.0
        )
        st.plotly_chart(
            premium_histogram(premiums, avg_allow), width="stretch"
        )
        share_below = (
            float((premiums < avg_allow).mean()) if len(premiums) else 0.0
        )
        _recommendation(
            f"About **{_fmt_pct(share_below)}** of active members chose a "
            "premium below the average allowance, so their leftover dollars go "
            "unused. Surfacing richer (higher-metal) plans to members with "
            "headroom would raise utilization and perceived benefit value."
        )


def view_cohorts(monthly: pd.DataFrame, members: pd.DataFrame) -> None:
    st.subheader("Cohort utilization over time")
    _synthetic_caption()
    st.markdown(
        "Each line is an **enrollment-month cohort**; the newest cohort is "
        "highlighted. Utilization climbs over a cohort's first few months as "
        "members finalize plans and reimbursements settle."
    )

    cohort = A.utilization_by_cohort_month(monthly, members)
    if cohort.empty:
        st.info("No monthly spend data available for cohort analysis.")
        return

    st.plotly_chart(cohort_lines(cohort), width="stretch")

    with st.expander("Cohort detail (utilization by month)"):
        show = cohort.pivot(
            index="enrollment_month", columns="month", values="utilization"
        )
        st.dataframe(
            show.style.format("{:.0%}", na_rep="—"), width="stretch"
        )

    # Steady-state (oldest cohort, latest month) vs. newest cohort's first month.
    latest = cohort["month"].max()
    first_cohort = cohort["enrollment_month"].min()
    last_cohort = cohort["enrollment_month"].max()
    steady = cohort[
        (cohort["enrollment_month"] == first_cohort)
        & (cohort["month"] == latest)
    ]
    newest_start = cohort[cohort["enrollment_month"] == last_cohort].iloc[0]
    steady_u = float(steady["utilization"].iloc[0]) if len(steady) else 0.0
    _recommendation(
        f"The earliest cohort ({first_cohort}) has settled near "
        f"{_fmt_pct(steady_u)} utilization, while the newest ({last_cohort}) "
        f"starts at {_fmt_pct(float(newest_start['utilization']))}. The gap is "
        "a predictable onboarding ramp, not churn — so judge a cohort's health "
        "only after ~3 months, and forecast spend using the ramp curve rather "
        "than the latest month alone."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Health-Benefits Spend & Utilization Analytics")
    st.markdown(
        "A portfolio demonstration of an ICHRA metrics layer — plan-selection "
        "funnel, allowance utilization, and cohort spend — surfaced as a BI "
        "dashboard."
    )

    if not os.path.exists(A.DEFAULT_MEMBERS_PATH):
        st.warning(
            "No data found. Run `python data_gen.py` once to create the "
            "synthetic dataset, then reload."
        )
        st.stop()

    employers, members, monthly = _load()
    clean, report = A.clean_members(members, employers)
    snapshot = A.month_snapshot(monthly, A.AS_OF_MONTH)

    with st.sidebar:
        st.header("View")
        view = st.radio(
            "Choose a view",
            [
                "Overview (KPIs)",
                "Plan-selection funnel",
                "Spend & utilization",
                "Cohort utilization",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            "### ⚠️ Synthetic data\n"
            "Every figure is a **seeded simulation** of an ICHRA benefits "
            "book — no real people, employers, or claims, and no protected "
            "health information."
        )
        st.caption(f"Reporting month: {A.AS_OF_MONTH}")
        st.caption(
            f"Source members: {report['rows_in']:,} · "
            f"Analysis members (post-QA): {report['rows_out']:,} "
            f"({report['pct_retained']}% retained)"
        )
        st.divider()
        st.markdown(
            "**Modeled like dbt:** `dim_employers`, `dim_members`, "
            "`fct_monthly_spend` → cleaned staging → funnel / KPI / spend / "
            "cohort marts in `analytics.py`."
        )
        st.divider()
        st.caption("Leo Chan · github.com/GoldenCrate")

    if view == "Overview (KPIs)":
        view_overview(clean, snapshot, report)
    elif view == "Plan-selection funnel":
        view_funnel(clean)
    elif view == "Spend & utilization":
        view_spend(snapshot, members, employers, clean)
    else:
        view_cohorts(monthly, members)


if __name__ == "__main__":
    main()
