"""
Hospital Patient No-Show Decision-Support Dashboard (CarePredict)
Clean, single-file Streamlit application for internship project defense.
Run locally via: streamlit run app.py
"""

from pathlib import Path
from datetime import date, timedelta
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from model_pipeline import add_booking_features, calibrate_probabilities

# Page configuration
st.set_page_config(
    page_title="CarePredict | Patient No-Show Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
VERSIONED_MODEL_PATH = MODEL_DIR / "model_v3.joblib"
BASELINE_MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_PATH = (
    VERSIONED_MODEL_PATH if VERSIONED_MODEL_PATH.exists() else BASELINE_MODEL_PATH
)
METADATA_PATH = MODEL_DIR / "model_v3_metadata.json"
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "test.parquet"
PREDICTIONS_PATH = BASE_DIR / "reports" / "historical_predictions.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource
def load_model_bundle(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        st.error(
            f"Model artifact not found: {path}. Run train_model.py first."
        )
        st.stop()
    return joblib.load(path)

bundle = load_model_bundle(str(MODEL_PATH))
preprocessor = bundle["preprocessor"]
model = bundle["model"]
calibrator = bundle["calibrator"]
threshold = float(bundle["optimal_threshold"])
modeling_features = bundle["modeling_features"]
metadata = None
if METADATA_PATH.is_file() and bundle.get("artifact_version") == 3:
    candidate_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if candidate_metadata.get("test"):
        metadata = candidate_metadata
if metadata is None:
    st.error(
        "Finalized historical prediction outputs are unavailable. Run train_model.py, "
        "review the frozen candidate, then run finalize_test.py once."
    )
    st.stop()
if metadata is not None and not DATA_PATH.is_file():
    st.error(f"Historical test data not found at {DATA_PATH}. Run preprocessing first.")
    st.stop()
if metadata is not None:
    if metadata["model_hash"] != sha256_file(MODEL_PATH):
        st.error(
            "Model metadata does not match the loaded artifact. "
            "Run the model-selection and finalization workflow again."
        )
        st.stop()
    expected_test_hash = metadata["source_hashes"].get(str(DATA_PATH))
    if expected_test_hash != sha256_file(DATA_PATH):
        st.error(
            "The historical test data differs from the evaluated artifact. "
            "Do not replace it; restore the recorded source data."
        )
        st.stop()
    if (
        not PREDICTIONS_PATH.is_file()
        or metadata["test"].get("prediction_artifact_hash")
        != sha256_file(PREDICTIONS_PATH)
    ):
        st.error(
            "Historical prediction outputs do not match the finalized evaluation. "
            "Restore the recorded outputs before opening the dashboard."
        )
        st.stop()

def classify_risk(probability):
    if probability < threshold:
        return "Low", "#34D399"
    if probability < 0.30:
        return "Medium", "#FBBF24"
    return "High", "#F87171"


def booking_context_summary(row) -> str:
    history_count = int(row["prev_appointment_count"])
    history_rate = float(row["prev_no_show_rate"])
    prior_history = (
        f"{history_count} prior visits · "
        f"{history_rate:.1%} observed prior no-show rate"
        if history_count
        else "No prior visit history"
    )
    return (
        f"- Lead time: {int(row['lead_time_days'])} days\n"
        f"- Age: {int(row['age'])} years\n"
        f"- Prior history: {prior_history}"
    )


# --- 2. LOAD & SCORE APPOINTMENTS ---
@st.cache_data
def get_scored_appointments(data_hash: str, prediction_hash: str):
    """Loads only finalized scores for the historical cohort."""
    if not DATA_PATH.exists():
        st.error(f"Historical test data not found at {DATA_PATH}. Run preprocessing first.")
        st.stop()
    df = add_booking_features(pd.read_parquet(DATA_PATH))
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    if predictions["appointment_id"].duplicated().any():
        raise ValueError("Finalized prediction artifact contains duplicate appointment IDs.")
    df = df.merge(
        predictions,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(df) != len(predictions) or df["no_show_prob"].isna().any():
        raise ValueError("Finalized predictions do not match the historical cohort.")
    df["prob_pct"] = (df["no_show_prob"] * 100).round(1)

    tiers, colors = zip(*(classify_risk(p) for p in df["no_show_prob"]))
    df["risk_level"] = tiers
    df["risk_color"] = colors

    df["patient_anon"] = df["patient_id"].apply(lambda pid: f"PT-***{str(pid)[-4:]}")
    df["schedule_date"] = df["appointment_day"].dt.date

    return df

df_appointments = get_scored_appointments(
    sha256_file(DATA_PATH),
    sha256_file(PREDICTIONS_PATH),
)
cohort_start = df_appointments["appointment_day"].min().date()
cohort_end = df_appointments["appointment_day"].max().date()

# --- 3. SESSION-ONLY REVIEW STATUS ---
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("CarePredict", icon=":material/health_and_safety:")

    selected_page = st.radio(
        "Dashboard",
        [
            "Historical planning simulation",
            "Individual appointment analysis",
            "Model performance & limitations",
        ],
        index=0,
    )

    st.badge("Historical holdout", icon=":material/history:", color="violet")
    st.caption(
        f"Test cohort: {cohort_start:%b %d}–{cohort_end:%b %d, %Y}"
    )
    st.caption(
        f"Model: {bundle.get('model_name', 'Original baseline')} · "
        f"{bundle.get('feature_set', 'baseline features')}"
    )
    st.caption(f"Model decision threshold: {threshold:.1%}")

    st.info(
        "No live hospital feed. No real-time schedule changes, messages, or cancellations.",
        title="Simulation only",
        icon=":material/info:",
    )

st.warning(
    "This dashboard scores a 2016 historical test cohort. Appointment times are synthetic, review marks are session-only, "
    "and no live scheduling or patient communication is performed.",
    title="Historical simulation — not live scheduling",
    icon=":material/warning:",
)

# ==============================================================================
# PAGE 1: HISTORICAL PLANNING SIMULATION
# ==============================================================================
if selected_page == "Historical planning simulation":
    available_dates = sorted(df_appointments["schedule_date"].unique())
    date_col, _ = st.columns([1, 2])
    with date_col:
        chosen_date = st.selectbox(
            "Historical appointment date",
            available_dates,
            format_func=lambda value: value.strftime("%B %d, %Y"),
        )

    day_df = (
        df_appointments.loc[df_appointments["schedule_date"] == chosen_date]
        .copy()
        .reset_index(drop=True)
    )
    total_appts = len(day_df)
    expected_no_shows = float(day_df["no_show_prob"].sum())
    no_show_rate = expected_no_shows / total_appts if total_appts else 0.0
    high_risk_count = int((day_df["risk_level"] == "High").sum())
    buffer_slots = int(np.floor(expected_no_shows * 0.70))

    daily_summary = (
        df_appointments.groupby("schedule_date")
        .agg(
            appointments=("appointment_id", "size"),
            expected_no_shows=("no_show_prob", "sum"),
            no_show_rate=("no_show_prob", "mean"),
            high_risk=("risk_level", lambda values: int((values == "High").sum())),
        )
        .sort_index()
    )
    daily_summary["buffer_slots"] = np.floor(
        daily_summary["expected_no_shows"] * 0.70
    ).astype(int)
    comparison_days = daily_summary.drop(index=chosen_date)
    historical_average = comparison_days.mean(numeric_only=True)

    def relative_delta(value, baseline):
        if not baseline or not np.isfinite(baseline):
            return "Baseline unavailable"
        return f"{(value / baseline - 1) * 100:+.1f}%"

    st.title(f"Historical planning simulation · {chosen_date:%B %d, %Y}")
    st.caption(
        f"Full historical test-cohort date: {total_appts:,} appointments. "
        "All comparisons use the mean of the other dates in this holdout cohort."
    )

    metrics = [
        (
            "Appointments",
            f"{total_appts:,}",
            relative_delta(total_appts, historical_average["appointments"]),
            "Historical test-cohort appointment rows for the selected date.",
        ),
        (
            "Expected no-shows",
            f"{expected_no_shows:,.1f}",
            relative_delta(expected_no_shows, historical_average["expected_no_shows"]),
            "Sum of calibrated appointment probabilities. This is an expectation, not a count of confirmed no-shows.",
        ),
        (
            "Predicted no-show rate",
            f"{no_show_rate:.1%}",
            relative_delta(no_show_rate, historical_average["no_show_rate"]),
            "Expected no-shows divided by all appointments for the selected date.",
        ),
        (
            "High-risk appointments",
            f"{high_risk_count:,}",
            relative_delta(high_risk_count, historical_average["high_risk"]),
            "Appointments with calibrated risk at or above 30%; display band only, not a clinical triage rule.",
        ),
        (
            "Buffer scenario (70%)",
            f"{buffer_slots:,}",
            relative_delta(buffer_slots, historical_average["buffer_slots"]),
            "Floor(expected no-shows × 70%). The 70% absorption assumption is illustrative, not validated or an instruction to overbook.",
        ),
    ]
    metric_columns = st.columns(5)
    for column, (label, value, delta, help_text) in zip(metric_columns, metrics):
        with column:
            st.metric(
                label,
                value,
                delta,
                delta_color="off",
                delta_description="vs other holdout dates",
                help=help_text,
                border=True,
                height="stretch",
            )

    st.caption(
        f"Method: expected no-shows = Σ calibrated probabilities ({expected_no_shows:,.1f}); "
        f"rate = expected no-shows ÷ appointments ({no_show_rate:.1%}); "
        f"buffer scenario = floor(expected no-shows × 0.70) ({buffer_slots}). "
        f"Display bands: Low < {threshold:.1%}, Medium {threshold:.1%}–<30%, High ≥30%. "
        "These bands are descriptive, not clinically validated."
    )

    trend_col, risk_col = st.columns(2)
    with trend_col:
        st.subheader("Appointments and expected no-shows by date")
        trend_dates = [value.strftime("%b %d, %Y") for value in daily_summary.index]
        trend_figure = go.Figure()
        trend_figure.add_trace(
            go.Bar(
                x=trend_dates,
                y=daily_summary["appointments"],
                name="Appointments",
                marker_color="#60A5FA",
                yaxis="y",
            )
        )
        trend_figure.add_trace(
            go.Scatter(
                x=trend_dates,
                y=daily_summary["expected_no_shows"],
                name="Expected no-shows",
                mode="lines+markers",
                line={"color": "#FBBF24", "width": 3},
                yaxis="y2",
            )
        )
        trend_figure.update_layout(
            height=330,
            margin={"l": 16, "r": 16, "t": 24, "b": 24},
            xaxis={"title": "Historical appointment date"},
            yaxis={"title": "Appointments"},
            yaxis2={
                "title": "Expected no-shows",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(
            trend_figure,
            alt="Daily historical appointment totals compared with expected no-shows",
            width="stretch",
        )

    with risk_col:
        st.subheader("Risk distribution for selected date")
        risk_order = ["Low", "Medium", "High"]
        risk_counts = day_df["risk_level"].value_counts().reindex(risk_order, fill_value=0)
        risk_percentages = risk_counts / total_appts * 100 if total_appts else risk_counts
        risk_colors = {"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"}
        risk_figure = go.Figure(
            go.Bar(
                x=risk_order,
                y=risk_counts.values,
                marker_color=[risk_colors[level] for level in risk_order],
                text=[f"{risk_percentages[level]:.1f}%" for level in risk_order],
                textposition="outside",
                customdata=risk_percentages.values,
                hovertemplate="%{x}: %{y:,} appointments (%{customdata:.1f}%)<extra></extra>",
            )
        )
        risk_figure.update_layout(
            height=330,
            margin={"l": 16, "r": 16, "t": 24, "b": 24},
            xaxis={"title": "Descriptive risk band"},
            yaxis={"title": "Appointments"},
            showlegend=False,
        )
        st.plotly_chart(
            risk_figure,
            alt="Appointment counts and percentages by descriptive risk band",
            width="stretch",
        )
        st.caption(
            f"Display bands: Low < {threshold:.1%}; Medium {threshold:.1%} to <30%; "
            "High ≥30%. They are not clinical triage categories."
        )
        probability_figure = go.Figure(
            go.Histogram(
                x=day_df["no_show_prob"] * 100,
                nbinsx=12,
                marker_color="#60A5FA",
                hovertemplate="%{x:.1f}% probability<br>%{y:,} appointments<extra></extra>",
            )
        )
        probability_figure.update_layout(
            height=260,
            margin={"l": 16, "r": 16, "t": 24, "b": 24},
            xaxis={"title": "Calibrated no-show probability (%)"},
            yaxis={"title": "Appointments"},
            showlegend=False,
        )
        st.plotly_chart(
            probability_figure,
            alt="Distribution of calibrated no-show probabilities for the selected historical date",
            width="stretch",
        )

    st.subheader("Historical score patterns by booking context")
    st.caption(
        "Average calibrated scores grouped by appointment lead time and prior-visit history. "
        "These descriptive associations are not causal explanations or individual feature attribution."
    )
    lead_order = [
        "Same-day (0d)",
        "1-3 days",
        "4-7 days",
        "8-14 days",
        "15-30 days",
        "31+ days",
    ]
    lead_summary = (
        day_df.groupby("lead_bucket", observed=True)
        .agg(
            appointments=("no_show_prob", "size"),
            average_score=("no_show_prob", "mean"),
        )
        .reindex(lead_order)
        .dropna(subset=["appointments"])
        .reset_index()
    )
    history_order = [
        "No prior visits",
        "Prior visits, no missed visits",
        "Some prior no-shows (<33%)",
        "Prior no-show rate ≥33%",
    ]
    history_group = np.select(
        [
            day_df["prev_appointment_count"].eq(0),
            day_df["prev_no_show_rate"].ge(0.33),
            day_df["prev_no_show_rate"].gt(0),
        ],
        [
            history_order[0],
            history_order[3],
            history_order[2],
        ],
        default=history_order[1],
    )
    history_summary = (
        day_df.assign(prior_visit_group=history_group)
        .groupby("prior_visit_group")
        .agg(
            appointments=("no_show_prob", "size"),
            average_score=("no_show_prob", "mean"),
        )
        .reindex(history_order)
        .dropna(subset=["appointments"])
        .reset_index()
    )

    lead_factor_col, history_factor_col = st.columns(2)
    with lead_factor_col:
        st.subheader("Average score by lead time")
        lead_figure = go.Figure(
            go.Bar(
                x=lead_summary["average_score"] * 100,
                y=lead_summary["lead_bucket"],
                orientation="h",
                marker_color="#60A5FA",
                customdata=lead_summary[["appointments"]].to_numpy(),
                hovertemplate=(
                    "%{y}<br>Average predicted probability: %{x:.1f}%"
                    "<br>Appointments: %{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )
        lead_figure.add_vline(
            x=threshold * 100,
            line_dash="dot",
            line_color="#FBBF24",
        )
        lead_figure.update_layout(
            height=330,
            margin={"l": 16, "r": 16, "t": 32, "b": 24},
            xaxis={"title": "Average predicted probability (%)"},
            yaxis={
                "categoryorder": "array",
                "categoryarray": lead_order,
            },
            showlegend=False,
        )
        st.plotly_chart(
            lead_figure,
            alt="Average predicted no-show probability by historical appointment lead-time group",
            width="stretch",
        )

    with history_factor_col:
        st.subheader("Average score by prior history")
        history_figure = go.Figure(
            go.Bar(
                x=history_summary["average_score"] * 100,
                y=history_summary["prior_visit_group"],
                orientation="h",
                marker_color="#60A5FA",
                customdata=history_summary[["appointments"]].to_numpy(),
                hovertemplate=(
                    "%{y}<br>Average predicted probability: %{x:.1f}%"
                    "<br>Appointments: %{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )
        history_figure.add_vline(
            x=threshold * 100,
            line_dash="dot",
            line_color="#FBBF24",
        )
        history_figure.update_layout(
            height=330,
            margin={"l": 16, "r": 16, "t": 32, "b": 24},
            xaxis={"title": "Average predicted probability (%)"},
            yaxis={
                "categoryorder": "array",
                "categoryarray": history_order,
            },
            showlegend=False,
        )
        st.plotly_chart(
            history_figure,
            alt="Average predicted no-show probability by historical prior-visit group",
            width="stretch",
        )
    st.caption(
        f"The dotted line marks the model decision threshold ({threshold:.1%}); "
        "bar values are cohort-average predicted probabilities."
    )

    st.subheader("Reminder call allocation")
    st.caption(
        "Choose a hypothetical call capacity. Appointments are ranked by model probability; "
        "expected coverage is the sum of probabilities among the selected visits."
    )
    ranked_all = day_df.sort_values("no_show_prob", ascending=False).reset_index(drop=True)
    max_calls = min(50, total_appts)
    call_capacity = st.slider(
        "Hypothetical call capacity",
        min_value=0,
        max_value=max_calls,
        value=min(30, max_calls),
        step=10,
        help="Planning control only. No calls are placed or scheduled by this dashboard.",
    )
    selected_for_calls = ranked_all.head(call_capacity)
    appointments_selected = len(selected_for_calls)
    expected_covered = float(selected_for_calls["no_show_prob"].sum())
    remaining_expected = max(0.0, expected_no_shows - expected_covered)
    coverage_pct = expected_covered / expected_no_shows if expected_no_shows else 0.0

    call_columns = st.columns(4)
    call_columns[0].metric(
        "Hypothetical calls",
        f"{call_capacity:,}",
        help="Hypothetical outreach capacity selected above.",
        border=True,
    )
    call_columns[1].metric(
        "Appointments prioritized",
        f"{appointments_selected:,}",
        help="Top-ranked historical appointments selected for the hypothetical call list; no calls are placed.",
        border=True,
    )
    call_columns[2].metric(
        "Expected no-shows covered",
        f"{expected_covered:,.1f}",
        help="Sum of calibrated probabilities among selected appointments; not observed call effectiveness.",
        border=True,
    )
    call_columns[3].metric(
        "Remaining expected no-shows",
        f"{remaining_expected:,.1f}",
        f"{coverage_pct:.1%} estimated coverage",
        delta_color="off",
        help="Selected-date expectation minus expected no-shows in the prioritized call list.",
        border=True,
    )

    capacity_values = [10, 20, 30, 40, 50]
    covered_by_capacity = [
        float(ranked_all.head(min(capacity, total_appts))["no_show_prob"].sum())
        for capacity in capacity_values
    ]
    capacity_coverage = (
        (np.asarray(covered_by_capacity) / expected_no_shows * 100).tolist()
        if expected_no_shows
        else [0.0] * len(capacity_values)
    )
    allocation_figure = go.Figure(
        go.Bar(
            x=[f"{capacity} calls" for capacity in capacity_values],
            y=covered_by_capacity,
            marker_color="#60A5FA",
            text=[f"{pct:.1f}% coverage" for pct in capacity_coverage],
            textposition="outside",
            customdata=capacity_coverage,
            hovertemplate="%{x}: %{y:.1f} expected no-shows (%{customdata:.1f}% of date expectation)<extra></extra>",
        )
    )
    allocation_figure.update_layout(
        height=300,
        margin={"l": 16, "r": 16, "t": 24, "b": 24},
        xaxis={"title": "Hypothetical call capacity"},
        yaxis={"title": "Expected no-shows covered"},
        showlegend=False,
    )
    st.plotly_chart(
        allocation_figure,
        alt="Expected no-shows covered at 10, 20, 30, 40, and 50 hypothetical calls",
        width="stretch",
    )
    st.caption(
        "Coverage is a model-based prioritization estimate, not the percentage of people who would answer "
        "or the effectiveness of an actual reminder call."
    )

    st.subheader("Prioritized historical appointments")
    st.caption(
        "Masked identifiers and synthetic appointment times are shown for demonstration. "
        "Review statuses last only for this app session."
    )
    filter_columns = st.columns([1.3, 1, 1.4])
    with filter_columns[0]:
        patient_search = st.text_input(
            "Search masked patient ID",
            placeholder="For example, PT-***7222",
            key="worklist_patient_search",
        ).strip()
    with filter_columns[1]:
        selected_risks = st.multiselect(
            "Risk category",
            risk_order,
            default=risk_order,
            key="worklist_risk_filter",
        )
    with filter_columns[2]:
        selected_clinics = st.multiselect(
            "Clinic",
            sorted(day_df["neighbourhood"].dropna().astype(str).unique()),
            default=[],
            help="Leave blank to include all clinics.",
            key="worklist_clinic_filter",
        )

    order_col, count_col = st.columns(2)
    with order_col:
        probability_order = st.selectbox(
            "Sort by prediction probability",
            ["Highest first", "Lowest first"],
            key="worklist_sort",
        )
    with count_col:
        display_count = st.selectbox(
            "Appointments to display",
            [25, 50, 100],
            index=0,
            key="worklist_display_count",
        )

    filtered_df = ranked_all.copy()
    if patient_search:
        filtered_df = filtered_df.loc[
            filtered_df["patient_anon"].str.contains(
                patient_search, case=False, regex=False, na=False
            )
        ]
    filtered_df = filtered_df.loc[filtered_df["risk_level"].isin(selected_risks)]
    if selected_clinics:
        filtered_df = filtered_df.loc[
            filtered_df["neighbourhood"].isin(selected_clinics)
        ]
    filtered_df = filtered_df.sort_values(
        "no_show_prob",
        ascending=probability_order == "Lowest first",
    )

    priority_by_id = {
        int(appointment_id): rank
        for rank, appointment_id in enumerate(ranked_all["appointment_id"], start=1)
    }
    filtered_df["call_priority"] = filtered_df["appointment_id"].map(priority_by_id)
    filtered_df["selected_for_call"] = filtered_df["call_priority"] <= call_capacity
    filtered_df["review_status"] = filtered_df["appointment_id"].map(
        lambda value: st.session_state.decisions.get(
            int(value), "Pending human review"
        )
    )
    export_columns = {
        "patient_anon": "Patient ID",
        "neighbourhood": "Clinic",
        "lead_time_days": "Lead time (days)",
        "age": "Age",
        "risk_level": "Risk category",
        "no_show_prob": "Predicted probability",
        "prev_appointment_count": "Prior visit count",
        "prev_no_show_rate": "Observed prior no-show rate",
        "call_priority": "Call priority",
        "selected_for_call": "In hypothetical call list",
        "review_status": "Review status",
    }
    export_data = filtered_df[list(export_columns)].rename(columns=export_columns)
    download_col, result_col = st.columns([1, 3])
    with download_col:
        st.download_button(
            "Export prioritized list",
            data=export_data.to_csv(index=False).encode("utf-8"),
            file_name=f"historical_prioritized_{chosen_date:%Y%m%d}.csv",
            mime="text/csv",
            icon=":material/download:",
            help="Exports only masked IDs and historical simulation fields.",
        )
    with result_col:
        st.caption(
            f"Showing {min(len(filtered_df), display_count):,} of {len(filtered_df):,} matching appointments. "
            f"First {call_capacity} appointments by probability are in the hypothetical call list."
        )

    displayed_df = filtered_df.head(display_count)
    for _, row in displayed_df.iterrows():
        appointment_id = int(row["appointment_id"])
        review_status = st.session_state.decisions.get(
            appointment_id, "Pending human review"
        )
        with st.container(border=True):
            patient_col, risk_col = st.columns([2, 1])
            with patient_col:
                st.markdown(f"**Patient ID:** `{row['patient_anon']}`")
            with risk_col:
                badge_color = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red",
                }[row["risk_level"]]
                st.badge(f"{row['risk_level']} risk · {row['prob_pct']:.1f}%", color=badge_color)

            detail_columns = st.columns(3)
            details = [
                ("Clinic", row["neighbourhood"]),
                ("Lead time", f"{int(row['lead_time_days'])} days"),
                ("Patient age", f"{int(row['age'])} years"),
            ]
            for column, (label, value) in zip(detail_columns, details):
                with column:
                    st.caption(label)
                    st.markdown(f"**{value}**")

            st.caption("Observed booking-time inputs · not patient-specific explanations")
            st.markdown(booking_context_summary(row))
            if review_status == "Pending human review":
                st.badge(
                    "Pending review · human verification required",
                    icon=":material/pending_actions:",
                    color="orange",
                )
            else:
                st.badge(review_status, icon=":material/verified:", color="blue")

            with st.expander("View appointment details"):
                st.write(
                    f"Historical date: {chosen_date:%B %d, %Y} · "
                    f"Prior appointments: {int(row['prev_appointment_count'])} · "
                    f"Prior observed no-show rate: {row['prev_no_show_rate']:.1%}"
                )

            action_columns = st.columns(
                2 if review_status == "Pending human review" else 3
            )
            if action_columns[0].button(
                "Call considered",
                key=f"review_call_{appointment_id}",
                help="Marks this visit reviewed for a call; no call is placed.",
            ):
                st.session_state.decisions[appointment_id] = "Reviewed: call considered"
                st.rerun()
            if action_columns[1].button(
                "No call needed",
                key=f"review_no_call_{appointment_id}",
                help="Marks this visit reviewed with no call; no appointment is changed.",
            ):
                st.session_state.decisions[appointment_id] = "Reviewed: no call"
                st.rerun()
            if review_status != "Pending human review" and action_columns[2].button(
                "Reset review",
                key=f"review_reset_{appointment_id}",
            ):
                del st.session_state.decisions[appointment_id]
                st.rerun()

    if filtered_df.empty:
        st.info("No appointments match these worklist filters.")

# ==============================================================================
# PAGE 2: INDIVIDUAL APPOINTMENT ANALYSIS
# ==============================================================================
elif selected_page == "Individual appointment analysis":
    st.title("Individual appointment analysis")
    st.caption(
        "Look up a masked visit from the historical holdout or score a hypothetical scenario. "
        "Neither workflow accesses current patients or schedules."
    )

    st.subheader("Historical visit lookup")
    patient_query = st.text_input(
        "Enter the exact masked patient ID",
        placeholder="For example, PT-***7222",
        help="Identifiers are masked and derived from the historical dataset. Multiple visits may match one masked ID.",
        key="historical_patient_lookup",
    ).strip()
    if patient_query:
        patient_matches = df_appointments.loc[
            df_appointments["patient_anon"].str.casefold() == patient_query.casefold()
        ].sort_values("appointment_day")
        if patient_matches.empty:
            st.info("No historical visits match that masked ID.")
        else:
            visit_rows = {
                int(row["appointment_id"]): row
                for _, row in patient_matches.iterrows()
            }
            visit_id = st.selectbox(
                "Matching historical visits",
                list(visit_rows),
                format_func=lambda value: (
                    f"{visit_rows[value]['appointment_day']:%b %d, %Y} · "
                    f"{visit_rows[value]['neighbourhood']}"
                ),
                key="historical_visit_choice",
            )
            visit = visit_rows[visit_id]
            visit_columns = st.columns(3)
            visit_columns[0].metric(
                "Predicted no-show probability",
                f"{visit['no_show_prob']:.1%}",
                help="Calibrated historical model estimate, not a confirmed outcome.",
                border=True,
            )
            visit_columns[1].metric(
                "Historical appointment date",
                f"{visit['appointment_day']:%b %d, %Y}",
                border=True,
            )
            visit_columns[2].metric("Clinic", str(visit["neighbourhood"]), border=True)
            st.badge(
                f"{visit['risk_level']} risk",
                color={"Low": "green", "Medium": "orange", "High": "red"}[
                    visit["risk_level"]
                ],
            )
            st.markdown("**Observed booking-time inputs**")
            st.markdown(booking_context_summary(visit))
            st.caption(
                "These raw values are model inputs, not per-appointment attributions or causal explanations. "
                "Global validation permutation importance is shown on the performance page."
            )
            with st.expander("How to interpret this score"):
                st.markdown(
                    f"The calibrated probability is **{visit['no_show_prob']:.1%}**. "
                    f"The display band is **{visit['risk_level']}** "
                    f"(Low < {threshold:.1%}; Medium {threshold:.1%} to <30%; High ≥30%). "
                    "The model does not provide per-visit feature attribution in this dashboard, and a probability is not a verdict."
                )

    with st.form("individual_form"):
        st.subheader("Hypothetical scenario inputs")
        st.caption("Use synthetic values only; these dates are not connected to the historical schedule.")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Synthetic patient age", 0, 115, 35)
            neighbourhoods = sorted(
                df_appointments["neighbourhood"].dropna().astype(str).unique()
            )
            neighbourhood = st.selectbox("Clinic / neighbourhood", neighbourhoods, index=0)

        with col2:
            booking_date = st.date_input(
                "Synthetic booking date",
                value=date(2016, 6, 3),
            )
            appointment_date = st.date_input(
                "Synthetic appointment date",
                value=date(2016, 6, 3) + timedelta(days=10),
            )
            booking_hour = st.slider("Synthetic booking hour", 6, 20, 9)

        with col3:
            prev_count = st.number_input("Prior completed appointments", 0, 50, 2)
            prev_missed = st.number_input(
                "Prior missed appointments",
                0,
                int(prev_count),
                min(1, int(prev_count)),
            )
            prev_rate = float(prev_missed / prev_count) if prev_count > 0 else 0.0
            st.caption(f"Scenario prior no-show rate: **{prev_rate:.1%}**")

        predict_btn = st.form_submit_button(
            "Score hypothetical scenario",
            type="primary",
            icon=":material/query_stats:",
        )

    if predict_btn:
        if appointment_date < booking_date:
            st.error("Error: Appointment date cannot be earlier than booking date.")
        else:
            lead_time = max(0, (appointment_date - booking_date).days)
            weekday = appointment_date.strftime("%A")

            if age <= 12:
                age_group = "0-12"
            elif age <= 24:
                age_group = "13-24"
            elif age <= 45:
                age_group = "25-45"
            elif age <= 65:
                age_group = "46-65"
            else:
                age_group = "66+"

            input_df = add_booking_features(
                pd.DataFrame(
                    [
                        {
                            "scheduled_day": pd.Timestamp(booking_date)
                            + pd.Timedelta(hours=booking_hour),
                            "appointment_day": pd.Timestamp(appointment_date),
                            "age": age,
                            "neighbourhood": neighbourhood,
                            "prev_appointment_count": prev_count,
                            "prev_no_show_rate": prev_rate,
                        }
                    ]
                )
            )

            X_trans = preprocessor.transform(input_df[modeling_features])
            raw_p = float(model.predict_proba(X_trans)[0, 1])
            cal_p = float(
                calibrate_probabilities(np.array([raw_p]), calibrator)[0]
            )

            tier, color = classify_risk(cal_p)

            st.subheader("Hypothetical scenario result")

            c_res1, c_res2 = st.columns([1, 1.5])
            with c_res1:
                st.metric(
                    "Calibrated no-show probability",
                    f"{cal_p:.1%}",
                    help="Model estimate for the synthetic inputs above.",
                    border=True,
                )
                st.badge(
                    f"{tier} risk",
                    color={"Low": "green", "Medium": "orange", "High": "red"}[tier],
                )

            with c_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(cal_p * 100, 1),
                    number={"suffix": "%", "font": {"color": color, "size": 36}},
                    title={"text": "Scenario probability", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#064E3B"},
                            {"range": [threshold * 100, 30], "color": "#78350F"},
                            {"range": [30, 100], "color": "#7F1D1D"},
                        ],
                        "threshold": {
                            "line": {"color": "#F87171", "width": 3},
                            "value": 30,
                        },
                    }
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(
                    fig_gauge,
                    alt="Calibrated probability for the hypothetical scenario",
                    width="stretch",
                )

            st.subheader("Scenario inputs")
            st.caption(
                "The model combines these values. They are not ranked by importance; "
                "the current model bundle does not provide per-visit feature attribution."
            )
            input_summary = pd.DataFrame(
                [
                    ("Booking lead time", f"{lead_time} days"),
                    ("Appointment weekday", weekday),
                    ("Patient age band", age_group),
                    ("Prior appointments", f"{int(prev_count)}; {prev_missed} missed"),
                    ("Prior no-show rate", f"{prev_rate:.1%}"),
                    ("Clinic", neighbourhood),
                    ("Booking hour", f"{booking_hour:02d}:00"),
                ],
                columns=["Model input", "Scenario value"],
            )
            st.dataframe(
                input_summary,
                hide_index=True,
                alt="Synthetic values supplied to the model",
            )

            st.warning(
                "This is a calibrated probability for synthetic inputs, not a verdict or a recommendation. "
                "It is not connected to a live patient or schedule.",
                title="Interpretation limits",
            )

# ==============================================================================
# PAGE 3: MODEL PERFORMANCE AND LIMITATIONS
# ==============================================================================
elif selected_page == "Model performance & limitations":
    st.title("Model performance & limitations")
    st.caption(
        f"Offline evaluation on the chronological 2016 holdout cohort "
        f"({len(df_appointments):,} appointments, {cohort_start:%b %d}–{cohort_end:%b %d, %Y}). "
        "These historical results do not establish current hospital performance."
    )

    if metadata is None:
        st.warning(
            "Final evaluation metadata is unavailable. Run train_model.py, review "
            "the frozen candidate, then run finalize_test.py once. No performance "
            "values are substituted.",
            title="No generated metrics available",
        )
    else:
        test_metrics = metadata["test"]["high_recall"]
        selected = metadata["selected_model"]
        st.caption(
            f"Selected: {selected['model']} · {selected['feature_set']} · "
            f"{metadata['calibration_method']} calibration · "
            f"threshold {selected['threshold']:.3f}. "
            "This candidate was scored once at finalization; earlier workflow versions had prior test evaluation."
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Precision",
            f"{test_metrics['precision']:.1%}",
            help="Share of high-recall policy alerts that were no-shows in this historical holdout.",
            border=True,
        )
        metric_columns[1].metric(
            "Recall",
            f"{test_metrics['recall']:.1%}",
            help="Share of observed no-shows detected at the validation-selected operating threshold.",
            border=True,
        )
        metric_columns[2].metric(
            "F1-score",
            f"{test_metrics['f1']:.3f}",
            help="Harmonic mean of precision and recall at the selected threshold.",
            border=True,
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("ROC-AUC", f"{test_metrics['roc_auc']:.4f}", border=True)
        metric_columns[1].metric("PR-AUC", f"{test_metrics['pr_auc']:.4f}", border=True)
        metric_columns[2].metric(
            "Brier score",
            f"{test_metrics['brier_score']:.4f}",
            help="Lower is better; evaluates probability accuracy.",
            border=True,
        )

        st.subheader("Locked-test confusion matrix")
        st.caption(
            f"Validation-selected threshold: {selected['threshold']:.3f} · "
            f"false positives: {test_metrics['fp']:,} · false negatives: {test_metrics['fn']:,}"
        )
        confusion_figure = go.Figure(
            go.Heatmap(
                z=[
                    [test_metrics["tn"], test_metrics["fp"]],
                    [test_metrics["fn"], test_metrics["tp"]],
                ],
                x=["Predicted attendance", "Predicted no-show"],
                y=["Observed attendance", "Observed no-show"],
                colorscale="Blues",
                text=[
                    [f"TN<br>{test_metrics['tn']:,}", f"FP<br>{test_metrics['fp']:,}"],
                    [f"FN<br>{test_metrics['fn']:,}", f"TP<br>{test_metrics['tp']:,}"],
                ],
                texttemplate="%{text}",
                showscale=False,
            )
        )
        confusion_figure.update_layout(
            height=280,
            margin={"l": 16, "r": 16, "t": 16, "b": 16},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(
            confusion_figure,
            alt="Confusion matrix counts for the selected model on the locked historical test cohort",
            width="stretch",
        )

        st.subheader("Validation model comparison")
        comparison = pd.DataFrame(metadata["validation"]["model_comparison"])
        comparison_columns = {
            "model": "Model",
            "feature_set": "Feature set",
            "configuration": "Configuration",
            "calibration_method": "Calibration",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier_score": "Brier",
            "log_loss": "Log loss",
            "ece_10": "ECE (10 bins)",
            "high_recall_f1": "High-recall F1",
            "high_recall_precision": "Precision at high recall",
            "high_recall_recall": "Recall",
            "same_day_precision": "Same-day precision",
            "same_day_recall": "Same-day recall",
            "same_day_n": "Same-day n",
            "high_recall_tn": "TN",
            "high_recall_fp": "False positives",
            "high_recall_fn": "False negatives",
            "high_recall_tp": "TP",
            "fit_seconds": "Training seconds",
            "inference_ms_per_1000": "Inference ms / 1k",
        }
        comparison_display = comparison[
            list(comparison_columns)
        ].rename(columns=comparison_columns)
        st.dataframe(
            comparison_display,
            hide_index=True,
            alt="Validation comparison of model families, feature sets, calibration, and high-recall outcomes",
        )
        st.caption(
            "May 31 is the model/configuration selection window. Preprocessing is fit on training data; "
            "May 30 fits calibration, June 1 selects thresholds, and June 2 is a fixed-policy audit. "
            "The saved baseline is included as-is and recalibrated on May 30."
        )

        curve_columns = st.columns(2)
        curves = metadata["validation"]["curves"]
        with curve_columns[0]:
            st.subheader("Validation precision–recall curve")
            pr_curve = curves["precision_recall"]
            pr_figure = go.Figure(
                go.Scatter(
                    x=pr_curve["recall"],
                    y=pr_curve["precision"],
                    mode="lines",
                    name="Selected model",
                    line={"color": "#60A5FA", "width": 3},
                )
            )
            pr_figure.update_layout(
                xaxis={"title": "Recall", "range": [0, 1]},
                yaxis={"title": "Precision", "range": [0, 1]},
                height=320,
                margin={"l": 16, "r": 16, "t": 16, "b": 16},
                showlegend=False,
            )
            st.plotly_chart(
                pr_figure,
                alt="Precision versus recall on the June 1 threshold-selection window",
                width="stretch",
            )

        with curve_columns[1]:
            st.subheader("Validation ROC curve")
            roc_curve_data = curves["roc"]
            roc_figure = go.Figure()
            roc_figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line={"dash": "dash", "color": "#94A3B8"},
                    name="Chance",
                )
            )
            roc_figure.add_trace(
                go.Scatter(
                    x=roc_curve_data["false_positive_rate"],
                    y=roc_curve_data["true_positive_rate"],
                    mode="lines",
                    name="Selected model",
                    line={"color": "#34D399", "width": 3},
                )
            )
            roc_figure.update_layout(
                xaxis={"title": "False-positive rate", "range": [0, 1]},
                yaxis={"title": "True-positive rate", "range": [0, 1]},
                height=320,
                margin={"l": 16, "r": 16, "t": 16, "b": 16},
                showlegend=False,
            )
            st.plotly_chart(
                roc_figure,
                alt="Receiver operating characteristic curve on the validation tuning window",
                width="stretch",
            )

        calibration_columns = st.columns(2)
        with calibration_columns[0]:
            st.subheader("Validation calibration comparison")
            calibration_curves = curves["calibration"]
            calibration_rows = metadata["validation"]["calibration_comparison"]
            selected_calibration_rows = [
                row
                for row in calibration_rows
                if row["model"] == selected["model"]
                and row["feature_set"] == selected["feature_set"]
            ]
            brier_by_method = {
                row["method"]: row["brier_score"]
                for row in selected_calibration_rows
            }
            calibration_figure = go.Figure()
            calibration_figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect calibration",
                    line={"dash": "dash", "color": "#94A3B8"},
                )
            )
            for method in ("uncalibrated", "isotonic", "sigmoid"):
                points = calibration_curves[method]
                calibration_figure.add_trace(
                    go.Scatter(
                        x=points["mean_predicted"],
                        y=points["observed_fraction"],
                        mode="lines+markers",
                        name=(
                            f"{method.title()} "
                            f"(Brier {brier_by_method[method]:.4f})"
                        ),
                    )
                )
            calibration_figure.update_layout(
                xaxis={"title": "Mean predicted probability", "range": [0, 1]},
                yaxis={"title": "Observed no-show fraction", "range": [0, 1]},
                height=340,
                margin={"l": 16, "r": 16, "t": 16, "b": 16},
                legend={"orientation": "h", "y": -0.3},
            )
            st.plotly_chart(
                calibration_figure,
                alt="Uncalibrated, isotonic, and sigmoid reliability curves using validation data",
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(selected_calibration_rows)[
                    ["method", "brier_score", "log_loss", "ece_10"]
                ].rename(
                    columns={
                        "method": "Calibration",
                        "brier_score": "Brier",
                        "log_loss": "Log loss",
                        "ece_10": "ECE (10 bins)",
                    }
                ),
                hide_index=True,
                alt="Validation Brier score, log loss, and expected calibration error by calibration method",
            )

        with calibration_columns[1]:
            st.subheader("Threshold trade-offs on validation")
            threshold_rows = metadata["validation"]["threshold_sweep"]
            threshold_frame = pd.DataFrame(threshold_rows)
            threshold_figure = go.Figure()
            for column, label in (
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
            ):
                threshold_figure.add_trace(
                    go.Scatter(
                        x=threshold_frame["threshold"],
                        y=threshold_frame[column],
                        mode="lines",
                        name=label,
                    )
                )
            threshold_figure.add_vline(
                x=selected["threshold"],
                line_dash="dash",
                line_color="#FBBF24",
                annotation_text="Selected high-recall threshold",
            )
            threshold_figure.update_layout(
                xaxis={"title": "Decision threshold"},
                yaxis={"title": "Metric", "range": [0, 1]},
                height=340,
                margin={"l": 16, "r": 16, "t": 32, "b": 16},
                legend={"orientation": "h", "y": -0.3},
            )
            st.plotly_chart(
                threshold_figure,
                alt="Precision, recall, and F1 across validation-selected thresholds",
                width="stretch",
            )

        st.subheader("Operating strategy comparison")
        validation_strategies = pd.DataFrame(
            metadata["validation"]["threshold_strategies"]
        )
        test_strategies = pd.DataFrame(metadata["test"]["strategies"])
        strategy_columns = [
            "strategy",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fp",
            "fn",
            "tn",
            "tp",
        ]
        strategy_columns = [
            name for name in strategy_columns if name in validation_strategies
        ]
        strategy_col1, strategy_col2 = st.columns(2)
        with strategy_col1:
            st.markdown("**Validation policy selection**")
            st.dataframe(
                validation_strategies[strategy_columns],
                hide_index=True,
                alt="Validation operating strategies: high recall, balanced F1, and daily review capacity",
            )
        with strategy_col2:
            st.markdown("**Locked-test outcomes**")
            test_columns = [
                name for name in strategy_columns if name in test_strategies
            ]
            st.dataframe(
                test_strategies[test_columns],
                hide_index=True,
                alt="Locked-test outcomes for the validation-selected operating strategies",
            )
        st.caption(
            "High-recall targets 90% recall when achievable. Balanced maximizes validation F1. "
            "Resource-efficient prioritizes at most 30 visits per day; that capacity is illustrative."
        )

        subgroup_data = pd.DataFrame(metadata["test"]["subgroups"])
        subgroup_columns = [
            "group_column",
            "group",
            "n",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        subgroup_columns = [
            name for name in subgroup_columns if name in subgroup_data
        ]
        st.subheader("Locked-test subgroup performance")
        subgroup_age, subgroup_lead, subgroup_clinic = st.columns(3)
        with subgroup_age:
            st.markdown("**By age band**")
            st.dataframe(
                subgroup_data.loc[
                    subgroup_data["group_column"] == "age_group",
                    subgroup_columns,
                ],
                hide_index=True,
                alt="Selected-model precision, recall, and confusion counts by age band on locked test data",
            )
        with subgroup_lead:
            st.markdown("**By appointment lead time**")
            st.dataframe(
                subgroup_data.loc[
                    subgroup_data["group_column"] == "lead_bucket",
                    subgroup_columns,
                ],
                hide_index=True,
                alt="Selected-model precision, recall, and confusion counts by lead-time group on locked test data",
            )
        with subgroup_clinic:
            st.markdown("**By neighbourhood**")
            st.dataframe(
                subgroup_data.loc[
                    subgroup_data["group_column"] == "neighbourhood",
                    subgroup_columns,
                ],
                hide_index=True,
                alt="Selected-model precision, recall, and confusion counts by neighbourhood on locked test data",
            )

        same_day = metadata["test"]["same_day"]
        validation_same_day = next(
            row
            for row in metadata["validation"]["subgroups"]
            if row["group_column"] == "lead_bucket"
            and row["group"] == "Same-day (0d)"
        )
        st.subheader("Same-day appointment investigation")
        st.caption(
            f"Same-day test cohort: {same_day['n']:,} appointments, "
            f"{same_day['no_shows']:,} observed no-shows "
            f"({same_day['no_show_rate']:.1%}). No separate same-day classifier is used."
        )
        same_day_table = pd.DataFrame(
            [
                {
                    "Model / evaluation split": f"{selected['model']} · June 2 audit",
                    "Appointments": validation_same_day["n"],
                    "Precision": validation_same_day["precision"],
                    "Recall": validation_same_day["recall"],
                    "TN": validation_same_day["tn"],
                    "FP": validation_same_day["fp"],
                    "FN": validation_same_day["fn"],
                    "TP": validation_same_day["tp"],
                },
                {
                    "Model / evaluation split": f"{selected['model']} · locked test",
                    "Appointments": same_day["n"],
                    "Precision": same_day["precision"],
                    "Recall": same_day["recall"],
                    "TN": same_day["tn"],
                    "FP": same_day["fp"],
                    "FN": same_day["fn"],
                    "TP": same_day["tp"],
                },
            ]
        )
        st.dataframe(
            same_day_table,
            hide_index=True,
            alt="Selected model same-day precision, recall, and confusion counts on validation audit and locked test",
        )
        validation_context = pd.DataFrame(
            metadata["validation"]["same_day_analysis"]
        )
        context_columns = [
            "booking_group",
            "appointments",
            "no_shows",
            "no_show_rate",
            "median_age",
            "mean_prior_appointments",
            "mean_prior_no_show_rate",
            "mean_predicted_probability",
            "missing_feature_cells",
        ]
        st.dataframe(
            validation_context[context_columns].rename(
                columns={
                    "booking_group": "Booking group",
                    "appointments": "Appointments",
                    "no_shows": "Observed no-shows",
                    "no_show_rate": "Observed no-show rate",
                    "median_age": "Median age",
                    "mean_prior_appointments": "Mean prior appointments",
                    "mean_prior_no_show_rate": "Mean prior no-show rate",
                    "mean_predicted_probability": "Mean predicted probability",
                    "missing_feature_cells": "Missing feature cells",
                }
            ),
            hide_index=True,
            alt="Validation booking-context summaries comparing same-day appointments with longer lead times",
        )

        importance = pd.DataFrame(
            metadata["validation"]["feature_importance"]
        ).sort_values("mean_average_precision_drop")
        importance_figure = go.Figure(
            go.Bar(
                x=importance["mean_average_precision_drop"],
                y=importance["feature"],
                orientation="h",
                marker_color="#60A5FA",
                error_x={"type": "data", "array": importance["std_average_precision_drop"]},
            )
        )
        importance_figure.update_layout(
            xaxis={"title": "Average-precision drop after shuffling"},
            yaxis={"title": "Feature"},
            height=max(300, 32 * len(importance)),
            margin={"l": 16, "r": 16, "t": 16, "b": 16},
            showlegend=False,
        )
        st.subheader("Global validation feature importance")
        st.caption(
            "Permutation importance is global and descriptive, not causal or per-appointment attribution."
        )
        st.plotly_chart(
            importance_figure,
            alt="Validation permutation importance measured as average-precision drop",
            width="stretch",
        )

        st.subheader("Limitations and data disclosure")
        for limitation in metadata["limitations"]:
            st.markdown(f"- {limitation}")
        st.caption(
            "Predictions are decision-support estimates only. Human review is required; "
            "the dashboard never cancels appointments or contacts patients."
        )
