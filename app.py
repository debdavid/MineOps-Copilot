import os
import numpy as np
import pandas as pd
import streamlit as st

from typing import TypedDict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix,
)

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# ============================================================
# 0. PAGE + DATA CONFIG
# ============================================================

st.set_page_config(
    page_title="VANTAGE | MineOps Copilot",
    layout="wide",
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

FEATURES = [
    "Type",
    "Air_Temp_K",
    "Process_Temp_K",
    "Rotational_Speed_rpm",
    "Torque_Nm",
    "Tool_Wear_min",
]

NUMERIC_FEATURES = [
    "Air_Temp_K",
    "Process_Temp_K",
    "Rotational_Speed_rpm",
    "Torque_Nm",
    "Tool_Wear_min",
]

TARGET = "Machine_Failure"

COLUMN_MAP = {
    "UDI": "UDI",
    "Product ID": "Product_ID",
    "Type": "Type",
    "Air temperature [K]": "Air_Temp_K",
    "Process temperature [K]": "Process_Temp_K",
    "Rotational speed [rpm]": "Rotational_Speed_rpm",
    "Torque [Nm]": "Torque_Nm",
    "Tool wear [min]": "Tool_Wear_min",
    "Machine failure": "Machine_Failure",
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
}

DISPLAY_NAMES = {
    "Air_Temp_K": "Air temperature",
    "Process_Temp_K": "Process temperature",
    "Rotational_Speed_rpm": "Rotational speed",
    "Torque_Nm": "Torque",
    "Tool_Wear_min": "Tool wear",
}

UNITS = {
    "Air_Temp_K": "K",
    "Process_Temp_K": "K",
    "Rotational_Speed_rpm": "rpm",
    "Torque_Nm": "Nm",
    "Tool_Wear_min": "min",
}


# ============================================================
# 1. DATA
# ============================================================

@st.cache_data
def load_full_dataset():
    """
    Prefer a local copy for demo reliability.
    """
    df = None

    for local_path in ["ai4i2020.csv", "ai4i2020_full.csv"]:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            break

    if df is None:
        df = pd.read_csv(DATA_URL)

    df = df.rename(columns=COLUMN_MAP)

    required = FEATURES + [TARGET, "UDI"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ============================================================
# 2. MODEL
# ============================================================

@st.cache_resource
def train_failure_model(df):
    """
    Random Forest model:
    - learns relationships from labelled data
    - avoids arbitrary hand-set feature weights

    Threshold:
    - chosen on validation data using F2
    - F2 weights recall more heavily than precision
    """

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Type"],
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    val_probability = model.predict_proba(X_val)[:, 1]

    threshold_rows = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        val_prediction = (
            val_probability >= threshold
        ).astype(int)

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_val,
                    val_prediction,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_val,
                    val_prediction,
                    zero_division=0,
                ),
                "f2": fbeta_score(
                    y_val,
                    val_prediction,
                    beta=2,
                    zero_division=0,
                ),
            }
        )

    threshold_table = pd.DataFrame(threshold_rows)

    best_row = threshold_table.loc[
        threshold_table["f2"].idxmax()
    ]

    alert_threshold = float(best_row["threshold"])

    test_probability = model.predict_proba(X_test)[:, 1]

    test_prediction = (
        test_probability >= alert_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_prediction,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "alert_threshold": alert_threshold,
        "precision": precision_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_test,
            test_prediction,
            beta=2,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_test,
            test_probability,
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }

    test_records = X_test.copy()
    test_records[TARGET] = y_test.values
    test_records["Model_Risk"] = test_probability
    test_records["UDI"] = df.loc[
        test_records.index,
        "UDI",
    ].values

    return model, metrics, threshold_table, test_records


# ============================================================
# 3. DEMO FLEET
# ============================================================

@st.cache_data
def build_demo_fleet(test_records):
    """
    Build a 100-record demonstration fleet.
    Historical failures are deliberately oversampled for demo visibility.
    """

    failed = test_records[
        test_records[TARGET] == 1
    ]

    healthy = test_records[
        test_records[TARGET] == 0
    ]

    n_failed = min(15, len(failed))
    n_healthy = 100 - n_failed

    demo = pd.concat(
        [
            failed.sample(
                n=n_failed,
                random_state=42,
            ),
            healthy.sample(
                n=n_healthy,
                random_state=42,
            ),
        ]
    )

    return (
        demo.sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


# ============================================================
# 4. TELEMETRY CONTEXT
# ============================================================

def percentile_rank(
    series: pd.Series,
    value: float,
) -> float:
    """
    Percentage of reference observations at or below current value.
    """
    return float(
        (series <= value).mean() * 100
    )


def relative_band(percentile: float) -> str:
    """
    Simple attention cue only.
    """
    if percentile >= 90 or percentile <= 10:
        return "Review signal"
    return "Typical"


def relative_position_text(
    percentile: float,
) -> str:
    """
    Plain-language comparison against the reference dataset.
    """
    if percentile >= 95:
        return "Well above typical range"
    if percentile >= 90:
        return "Above typical range"
    if percentile <= 5:
        return "Well below typical range"
    if percentile <= 10:
        return "Below typical range"
    return "Within typical range"


def build_telemetry_context(
    full_df: pd.DataFrame,
    selected_row: pd.Series,
) -> pd.DataFrame:

    rows = []

    for feature in NUMERIC_FEATURES:
        value = float(
            selected_row[feature]
        )

        percentile = percentile_rank(
            full_df[feature],
            value,
        )

        q10 = float(
            full_df[feature].quantile(0.10)
        )

        q90 = float(
            full_df[feature].quantile(0.90)
        )

        rows.append(
            {
                "Signal":
                    DISPLAY_NAMES[feature],

                "Current":
                    f"{value:.1f} {UNITS[feature]}",

                "Typical range":
                    f"{q10:.1f}–{q90:.1f} "
                    f"{UNITS[feature]}",

                "Position":
                    relative_position_text(
                        percentile
                    ),

                "Flag":
                    relative_band(
                        percentile
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. GENERATIVE-AI EXPLANATION LAYER
# ============================================================

load_dotenv()

GROQ_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]


def invoke_llm_with_fallback(
    prompt: str,
):
    """
    Optional explanation layer.
    The quantitative dashboard keeps working if Groq fails.
    """

    key = os.getenv("GROQ_API_KEY")

    if not key:
        return (
            None,
            "No GROQ_API_KEY configured",
        )

    last_error = None

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0.1,
                groq_api_key=key,
            )

            response = llm.invoke(
                prompt
            )

            return (
                response.content,
                model_name,
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

    return None, last_error


class AgentState(TypedDict):
    evidence: str
    explanation: str
    handover: str


def pass_evidence(
    state: AgentState,
):
    return {
        "evidence":
            state["evidence"]
    }


def explain_risk(
    state: AgentState,
):
    prompt = f"""
You are supporting a reliability engineering team.

IMPORTANT BOUNDARIES:
- A quantitative ML model has already created the model risk score.
- Do NOT independently diagnose a mechanical failure.
- Do NOT invent safety limits, engineering thresholds, repair costs,
  isolation procedures or financial savings.
- Statistical reference ranges are not engineering safety limits.
- Final maintenance decisions belong to authorised reliability /
  maintenance personnel.

MODEL EVIDENCE:
{state['evidence']}

Write for an operational user.

OUTPUT:
WHY THIS ASSET WAS FLAGGED:
Explain the model status in plain language.

NOTABLE TELEMETRY:
Mention only the readings that stand out most strongly relative to the
reference data.

WHAT HAPPENS NEXT:
Recommend either continued monitoring or reliability review, based on
the model status.
"""

    content, error = (
        invoke_llm_with_fallback(
            prompt
        )
    )

    if content is not None:
        return {
            "explanation": content
        }

    return {
        "explanation":
            "The generative-AI explanation layer is unavailable. "
            "The quantitative model remains operational. "
            f"Technical note: {error}"
    }


def draft_handover(
    state: AgentState,
):
    prompt = f"""
Convert this explanation into a concise shift-handover note.

BOUNDARIES:
- Do NOT invent safety procedures.
- Do NOT order equipment isolation.
- Do NOT invent financial values.
- Final action belongs to authorised reliability / maintenance staff.

EXPLANATION:
{state['explanation']}

FORMAT:
STATUS:
WHY:
NEXT STEP:
DECISION OWNER:
"""

    content, _ = (
        invoke_llm_with_fallback(
            prompt
        )
    )

    if content is not None:
        return {
            "handover": content
        }

    return {
        "handover":
            "STATUS: Review selected asset status.\n"
            "WHY: Refer to model score and current telemetry context.\n"
            "NEXT STEP: Follow the dashboard recommendation and approved "
            "site procedures.\n"
            "DECISION OWNER: Authorised reliability / maintenance team."
    }


workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "evidence",
    pass_evidence,
)

workflow.add_node(
    "explanation",
    explain_risk,
)

workflow.add_node(
    "handover",
    draft_handover,
)

workflow.add_edge(
    START,
    "evidence",
)

workflow.add_edge(
    "evidence",
    "explanation",
)

workflow.add_edge(
    "explanation",
    "handover",
)

workflow.add_edge(
    "handover",
    END,
)

app_engine = workflow.compile()


# ============================================================
# 6. PREPARE MODEL + DEMO DATA
# ============================================================

full_df = load_full_dataset()

(
    model,
    metrics,
    threshold_table,
    test_records,
) = train_failure_model(
    full_df
)

demo_df = build_demo_fleet(
    test_records
)

demo_df["Risk_Priority_Score"] = (
    demo_df["Model_Risk"] * 100
).round(1)

demo_df["Status"] = np.where(
    demo_df["Model_Risk"]
    >= metrics["alert_threshold"],
    "Review Required",
    "Monitor",
)

demo_df = demo_df.sort_values(
    by="Model_Risk",
    ascending=False,
).reset_index(drop=True)




# ============================================================
# 7. APP NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    ["Operations Dashboard", "Reliability Workspace"],
)

st.sidebar.caption(
    "VANTAGE | MineOps Copilot"
)


# ============================================================
# 8. OPERATIONS DASHBOARD
# ============================================================

if page == "Operations Dashboard":

    st.title(
        "VANTAGE | MineOps Copilot"
    )

    st.subheader(
        "Fleet Overview"
    )

    review_df = demo_df[
        demo_df["Status"] == "Review Required"
    ]

    monitor_df = demo_df[
        demo_df["Status"] == "Monitor"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Assets monitored",
        len(demo_df),
    )

    c2.metric(
        "Require review",
        len(review_df),
    )

    c3.metric(
        "Continue monitoring",
        len(monitor_df),
    )

    if len(review_df) > 0:
        st.warning(
            f"{len(review_df)} assets require reliability review."
        )
    else:
        st.success(
            "No assets currently require reliability review."
        )

    st.subheader(
        "Priority Assets"
    )

    priority_table = (
        demo_df[
            [
                "UDI",
                "Risk_Priority_Score",
                "Status",
            ]
        ]
        .head(10)
        .copy()
    )

    priority_table.columns = [
        "Asset ID",
        "Risk priority score (%)",
        "Status",
    ]

    st.dataframe(
        priority_table,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader(
        "Asset Status"
    )

    asset_options = demo_df.index.tolist()

    selected_index = st.selectbox(
        "Select asset:",
        options=asset_options,
        format_func=lambda i: str(demo_df.loc[i, "UDI"]),
        key="dashboard_asset_index",
    )

    selected_row = demo_df.loc[
        selected_index
    ].copy()

    selected_udi = selected_row["UDI"]

    risk_score = float(
        selected_row[
            "Risk_Priority_Score"
        ]
    )

    threshold_percent = (
        metrics["alert_threshold"] * 100
    )

    status = selected_row["Status"]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Risk priority score",
        f"{risk_score:.1f}%",
    )

    r2.metric(
        "Review threshold",
        f"{threshold_percent:.1f}%",
    )

    r3.metric(
        "Status",
        status,
    )

    if status == "Review Required":
        st.warning(
            "Reliability review required."
        )
    else:
        st.success(
            "Monitor"
        )

    st.markdown(
        "### Telemetry"
    )

    st.caption(
        f"Asset {selected_udi}"
    )

    telemetry_context = build_telemetry_context(
        full_df=full_df,
        selected_row=selected_row,
    )

    st.table(
        telemetry_context.set_index(
            "Signal"
        )
    )

    with st.popover(
        "What does 'Typical range' mean?"
    ):
        st.write(
            "This shows where most readings in the reference dataset usually fall. "
            "A reading outside this range is unusual and may deserve attention, "
            "but it does not automatically mean the equipment is unsafe or failing."
        )


# ============================================================
# 9. RELIABILITY WORKSPACE
# ============================================================

else:

    st.title(
        "Reliability Workspace"
    )

    review_df = demo_df[
        demo_df["Status"] == "Review Required"
    ].copy()

    if len(review_df) == 0:
        st.success(
            "No assets currently require reliability review."
        )
        st.stop()

    # --------------------------------------------------------
    # 9A. SELECT ASSET FOR ACTION
    # --------------------------------------------------------

    st.subheader(
        "Asset Action"
    )

    review_asset_options = (
        review_df.index.tolist()
    )

    selected_review_index = st.selectbox(
        "Select asset:",
        options=review_asset_options,
        format_func=lambda i: (
            f"{review_df.loc[i, 'UDI']}  |  "
            f"{review_df.loc[i, 'Risk_Priority_Score']:.1f}%"
        ),
        key="workspace_asset_index",
    )

    review_row = review_df.loc[
        selected_review_index
    ].copy()

    review_udi = review_row["UDI"]

    review_risk_score = float(
        review_row[
            "Risk_Priority_Score"
        ]
    )

    review_threshold = (
        metrics["alert_threshold"] * 100
    )

    review_telemetry = build_telemetry_context(
        full_df=full_df,
        selected_row=review_row,
    )

    # Show only concise context needed to act, not a duplicate dashboard.
    st.caption(
        f"Asset {review_udi} | Risk priority score {review_risk_score:.1f}%"
    )

    # --------------------------------------------------------
    # 9B. AI MAINTENANCE RECOMMENDATION
    # --------------------------------------------------------

    st.markdown(
        "### Maintenance Recommendation"
    )

    if st.button(
        "Generate Recommendation",
        type="primary",
        key="generate_recommendation",
    ):

        telemetry_text = (
            review_telemetry.to_string(
                index=False
            )
        )

        recommendation_prompt = f"""
You are supporting a mining reliability team.

Use ONLY the evidence below.

ASSET:
Asset ID: {review_udi}
Risk priority score: {review_risk_score:.1f}%
Review threshold: {review_threshold:.1f}%
Status: Review Required

TELEMETRY CONTEXT:
{telemetry_text}

BOUNDARIES:
- Do not diagnose a specific failure mode unless the evidence supports it.
- Do not invent OEM limits, safety limits, repair costs or shutdown requirements.
- Do not authorise isolation or maintenance.
- Recommendations must remain decision support for authorised reliability personnel.

Produce a concise operational recommendation with exactly these headings:

PRIORITY:
Choose High, Medium, or Low based on the evidence supplied.

RECOMMENDED ACTION:
State the next reasonable reliability action, such as inspect, trend, verify, or schedule review.

FOCUS AREAS:
Name the telemetry signals that deserve attention.

RATIONALE:
Explain briefly why this asset should be reviewed.

QUESTIONS FOR ENGINEER:
List 2-3 checks or questions the reliability engineer should consider before deciding.

DECISION OWNER:
State that final action sits with authorised reliability / maintenance personnel.
"""

        recommendation, error = (
            invoke_llm_with_fallback(
                recommendation_prompt
            )
        )

        if recommendation is not None:
            st.session_state["maintenance_recommendation"] = recommendation
            st.session_state["maintenance_recommendation_asset"] = review_udi
        else:
            st.session_state["maintenance_recommendation"] = None
            st.session_state["maintenance_recommendation_error"] = str(error)

    if st.session_state.get("maintenance_recommendation"):
        if st.session_state.get("maintenance_recommendation_asset") == review_udi:
            st.info(
                st.session_state["maintenance_recommendation"]
            )

    if (
        st.session_state.get("maintenance_recommendation") is None
        and st.session_state.get("maintenance_recommendation_error")
    ):
        st.error(
            "Recommendation service unavailable."
        )
        st.caption(
            st.session_state["maintenance_recommendation_error"]
        )

    # --------------------------------------------------------
    # 9C. DRAFT WORK ORDER
    # --------------------------------------------------------

    st.markdown(
        "### Draft Work Order"
    )

    if st.button(
        "Generate Draft Work Order",
        key="generate_work_order",
    ):

        prior_recommendation = st.session_state.get(
            "maintenance_recommendation",
            ""
        )

        work_order_prompt = f"""
You are preparing a draft maintenance work-order description for review.

ASSET:
Asset ID: {review_udi}
Risk priority score: {review_risk_score:.1f}%

AI MAINTENANCE RECOMMENDATION:
{prior_recommendation}

TELEMETRY:
{review_telemetry.to_string(index=False)}

BOUNDARIES:
- This is a draft only.
- Do not invent a failure diagnosis.
- Do not invent parts, costs, labour hours, shutdown duration, safety instructions or isolation requirements.
- Do not authorise work.
- Use concise professional maintenance language.

FORMAT:
ASSET:
REASON FOR REVIEW:
SUGGESTED INSPECTION SCOPE:
TELEMETRY TO VERIFY:
APPROVAL STATUS: Draft — reliability / maintenance approval required
"""

        work_order, error = (
            invoke_llm_with_fallback(
                work_order_prompt
            )
        )

        if work_order is not None:
            st.session_state["draft_work_order"] = work_order
            st.session_state["draft_work_order_asset"] = review_udi
        else:
            st.session_state["draft_work_order"] = None
            st.session_state["draft_work_order_error"] = str(error)

    if st.session_state.get("draft_work_order"):
        if st.session_state.get("draft_work_order_asset") == review_udi:
            st.success(
                st.session_state["draft_work_order"]
            )

    if (
        st.session_state.get("draft_work_order") is None
        and st.session_state.get("draft_work_order_error")
    ):
        st.error(
            "Draft work-order service unavailable."
        )
        st.caption(
            st.session_state["draft_work_order_error"]
        )

    # --------------------------------------------------------
    # 9D. SUPERVISOR FLEET HANDOVER
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Supervisor Handover"
    )

    st.caption(
        f"{len(review_df)} assets currently require review."
    )

    if st.button(
        "Generate Fleet Handover",
        key="generate_fleet_handover",
    ):

        fleet_rows = []

        for _, row in review_df.sort_values(
            "Risk_Priority_Score",
            ascending=False,
        ).iterrows():

            telemetry = build_telemetry_context(
                full_df=full_df,
                selected_row=row,
            )

            notable = telemetry[
                telemetry["Flag"] == "Review signal"
            ]

            notable_text = (
                ", ".join(
                    notable["Signal"].tolist()
                )
                if len(notable) > 0
                else "no single telemetry signal outside the typical range"
            )

            fleet_rows.append(
                f"Asset {row['UDI']}: "
                f"risk priority score {row['Risk_Priority_Score']:.1f}%; "
                f"notable signals: {notable_text}."
            )

        fleet_evidence = "\n".join(
            fleet_rows
        )

        fleet_prompt = f"""
You are preparing a supervisor shift handover for a mining reliability team.

CURRENT REVIEW QUEUE:
{fleet_evidence}

BOUNDARIES:
- Do not invent failures, safety incidents, repair costs, work orders or engineering limits.
- Do not authorise maintenance or isolation.
- Prioritise the highest-risk assets first.

Write exactly:

FLEET SUMMARY:
Two concise sentences summarising the current review workload.

TOP PRIORITIES:
List the top three assets in priority order with the main reason each deserves attention.

NEXT SHIFT FOCUS:
One concise sentence stating what the incoming shift should focus on.

DECISION OWNER:
Authorised reliability / maintenance personnel.
"""

        fleet_summary, error = (
            invoke_llm_with_fallback(
                fleet_prompt
            )
        )

        if fleet_summary is not None:
            st.session_state["fleet_handover"] = fleet_summary
        else:
            st.session_state["fleet_handover"] = None
            st.session_state["fleet_handover_error"] = str(error)

    if st.session_state.get("fleet_handover"):
        st.success(
            st.session_state["fleet_handover"]
        )

    if (
        st.session_state.get("fleet_handover") is None
        and st.session_state.get("fleet_handover_error")
    ):
        st.error(
            "Fleet handover service unavailable."
        )
        st.caption(
            st.session_state["fleet_handover_error"]
        )
