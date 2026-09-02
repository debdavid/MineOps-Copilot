
"""
VANTAGE v3 — Operations-Facing Decision Support Prototype

Design goals:
- Put the operational decision first.
- Keep technical/model details available but secondary.
- Avoid presenting statistical reference ranges as engineering safety limits.
- Keep the quantitative model separate from the generative-AI explanation layer.
- Keep the final maintenance decision with authorised reliability / maintenance personnel.

Dataset:
- UCI AI4I 2020 synthetic predictive-maintenance dataset.
- This is NOT real mine telemetry.
"""

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
    page_title="VANTAGE | Operations Intelligence",
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
# 1. DATA LOADING
# ============================================================

@st.cache_data
def load_full_dataset():
    """
    Prefer a local copy for interview reliability.
    Falls back to UCI only if the local file is unavailable.
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
# 2. MODEL TRAINING + VALIDATION
# ============================================================

@st.cache_resource
def train_failure_model(df):
    """
    Random Forest model:
    - learns nonlinear interactions from labelled historical examples
    - avoids arbitrary manual 40/40/20 weighting

    Data split:
    - 60% train
    - 20% validation
    - 20% test

    Threshold:
    - selected on validation data using F2 score
    - F2 gives more weight to recall than precision
    - production threshold would still need site-specific calibration
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
        transformers=[
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
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    # -------------------------
    # Choose alert threshold
    # -------------------------
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

    # -------------------------
    # Test performance
    # -------------------------
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
    Build a 100-record interview/demo fleet from held-out examples.

    Historical failures are deliberately oversampled to make the demo
    visually useful. This is NOT representative of a real fleet's
    natural failure prevalence.
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
# 4. REFERENCE CONTEXT FOR TELEMETRY
# ============================================================

def percentile_rank(series: pd.Series, value: float) -> float:
    """
    Percentage of reference observations at or below current value.
    """
    return float((series <= value).mean() * 100)


def reference_interpretation(percentile: float) -> str:
    """
    Statistical context only — NOT an engineering safety classification.
    """
    if percentile >= 90:
        return "Unusually high"
    if percentile <= 10:
        return "Unusually low"
    return "Within typical reference range"


def build_telemetry_context(
    full_df: pd.DataFrame,
    selected_row: pd.Series,
) -> pd.DataFrame:

    rows = []

    for feature in NUMERIC_FEATURES:
        value = float(selected_row[feature])
        pctl = percentile_rank(
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
                "Signal": DISPLAY_NAMES[feature],
                "Current reading": (
                    f"{value:.1f} {UNITS[feature]}"
                ),
                "Typical reference range*": (
                    f"{q10:.1f}–{q90:.1f} "
                    f"{UNITS[feature]}"
                ),
                "Relative position": (
                    f"Higher than {pctl:.0f}% "
                    "of reference observations"
                ),
                "Context": reference_interpretation(
                    pctl
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. LLM EXPLANATION LAYER
# ============================================================

load_dotenv()

GROQ_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]


def invoke_llm_with_fallback(prompt: str):
    """
    LLM is optional.
    The quantitative model/dashboard remain usable if Groq fails.
    """
    key = os.getenv("GROQ_API_KEY")

    if not key:
        return None, "No GROQ_API_KEY configured"

    last_error = None

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0.1,
                groq_api_key=key,
            )

            response = llm.invoke(prompt)

            return response.content, model_name

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

    return None, last_error


class AgentState(TypedDict):
    evidence: str
    technical_explanation: str
    handover_note: str


def evidence_router(state: AgentState):
    return {
        "evidence": state["evidence"]
    }


def reliability_explanation(
    state: AgentState,
):
    """
    LLM explains evidence.
    It does NOT independently diagnose the asset.
    """

    prompt = f"""
You are supporting a reliability engineering team.

IMPORTANT:
- A quantitative ML model has already produced the model risk score.
- Do NOT claim you independently diagnosed a mechanical failure.
- Do NOT invent safety limits, repair costs, isolation procedures,
  engineering thresholds or financial savings.
- Statistical reference ranges are NOT engineering safety limits.
- Final maintenance decisions belong to authorised reliability /
  maintenance personnel.

MODEL EVIDENCE:
{state['evidence']}

Write in plain operational language.

OUTPUT:
RISK SUMMARY:
Explain whether the asset is currently above or below the model review
threshold.

TELEMETRY CONTEXT:
Summarise only the most notable current telemetry observations.

NEXT STEP:
Recommend continued monitoring OR reliability review depending on the
model status, with final action determined according to approved site
procedures.
"""

    content, error = invoke_llm_with_fallback(
        prompt
    )

    if content is not None:
        return {
            "technical_explanation": content
        }

    return {
        "technical_explanation":
        "The generative-AI explanation layer is unavailable. "
        "The quantitative risk model remains operational. "
        f"Technical note: {error}"
    }


def operations_communication(
    state: AgentState,
):
    """
    Draft shift-handover communication.
    No autonomous maintenance instruction.
    """

    prompt = f"""
Turn the following reliability explanation into a concise shift-
handover note.

BOUNDARIES:
- Do NOT invent safety limits or safety procedures.
- Do NOT order isolation unless already authorised.
- Do NOT invent financial values.
- Final maintenance decision belongs to authorised reliability /
  maintenance personnel.

EXPLANATION:
{state['technical_explanation']}

FORMAT:
ALERT:
EVIDENCE:
RECOMMENDED NEXT STEP:
DECISION OWNER:
"""

    content, _ = invoke_llm_with_fallback(
        prompt
    )

    if content is not None:
        return {
            "handover_note": content
        }

    return {
        "handover_note":
        "ALERT: Review the selected asset status.\n"
        "EVIDENCE: Refer to the model risk score and current telemetry.\n"
        "RECOMMENDED NEXT STEP: Follow the dashboard status and approved "
        "site procedures.\n"
        "DECISION OWNER: Authorised reliability / maintenance team."
    }


workflow = StateGraph(AgentState)

workflow.add_node(
    "evidence",
    evidence_router,
)

workflow.add_node(
    "reliability_explainer",
    reliability_explanation,
)

workflow.add_node(
    "handover_writer",
    operations_communication,
)

workflow.add_edge(
    START,
    "evidence",
)

workflow.add_edge(
    "evidence",
    "reliability_explainer",
)

workflow.add_edge(
    "reliability_explainer",
    "handover_writer",
)

workflow.add_edge(
    "handover_writer",
    END,
)

app_engine = workflow.compile()


# ============================================================
# 6. PREPARE DATA + MODEL
# ============================================================

full_df = load_full_dataset()

(
    model,
    metrics,
    threshold_table,
    test_records,
) = train_failure_model(full_df)

demo_df = build_demo_fleet(
    test_records
)

demo_df["Model_Risk_Score"] = (
    demo_df["Model_Risk"] * 100
).round(1)

demo_df["Status"] = np.where(
    demo_df["Model_Risk"]
    >= metrics["alert_threshold"],
    "Review Required",
    "Monitor",
)

demo_df = demo_df.sort_values(
    "Model_Risk",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# 7. OPERATIONS-FACING DASHBOARD
# ============================================================

st.title(
    "VANTAGE: Operations Intelligence"
)

st.caption(
    "Predictive-maintenance decision-support prototype | "
    "Synthetic AI4I industrial data | "
    "Human reliability judgement retained"
)

st.info(
    "Prototype scope: this application uses synthetic industrial "
    "predictive-maintenance data, not live mine telemetry. "
    "Model scores support prioritisation; they are not engineering "
    "safety limits or autonomous maintenance decisions."
)


# ------------------------------------------------------------
# 7A. FLEET STATUS
# ------------------------------------------------------------

st.subheader(
    "Fleet Status"
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


# ------------------------------------------------------------
# 7B. PRIORITY ASSETS
# ------------------------------------------------------------

st.subheader(
    "Priority Assets"
)

priority_table = (
    demo_df[
        [
            "UDI",
            "Model_Risk_Score",
            "Status",
        ]
    ]
    .head(10)
    .copy()
)

priority_table.columns = [
    "Asset ID",
    "Model risk score (%)",
    "Status",
]

st.dataframe(
    priority_table,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Assets are ranked by model risk score so reliability teams can "
    "focus attention where the model indicates the greatest relative "
    "risk."
)


# ------------------------------------------------------------
# 7C. SELECT ASSET
# ------------------------------------------------------------

st.divider()

st.subheader(
    "Asset Review"
)

selected_udi = st.selectbox(
    "Select asset:",
    demo_df["UDI"].tolist(),
)

selected_row = demo_df[
    demo_df["UDI"] == selected_udi
].iloc[0]

risk_score = float(
    selected_row["Model_Risk_Score"]
)

threshold_percent = (
    metrics["alert_threshold"] * 100
)

status = selected_row["Status"]

a1, a2, a3 = st.columns(3)

a1.metric(
    "Model risk score",
    f"{risk_score:.1f}%",
)

a2.metric(
    "Review threshold",
    f"{threshold_percent:.1f}%",
)

a3.metric(
    "Decision-support status",
    status,
)

if status == "Review Required":
    st.warning(
        "This asset is above the model review threshold. "
        "A reliability / maintenance review is recommended before "
        "any consequential action is taken."
    )
else:
    st.success(
        "This asset is below the model review threshold. "
        "Continue monitoring unless site procedures or engineering "
        "judgement indicate otherwise."
    )


# ------------------------------------------------------------
# 7D. TELEMETRY CONTEXT
# ------------------------------------------------------------

st.markdown(
    "### Current Telemetry Context"
)

telemetry_context = build_telemetry_context(
    full_df,
    selected_row,
)

st.dataframe(
    telemetry_context,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "*Typical reference range = 10th–90th percentile of the synthetic "
    "reference dataset. This is statistical context only and must not "
    "be treated as an OEM limit, Whitehaven operating limit or safety "
    "threshold."
)


# ============================================================
# 8. AI-ASSISTED EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "AI-Assisted Reliability Brief"
)

st.write(
    "The quantitative model creates the risk signal. "
    "The LLM explains the evidence and drafts a handover note. "
    "It does not authorise maintenance action."
)

if st.button(
    "Generate Reliability Review Brief",
    type="primary",
):

    telemetry_text = telemetry_context.to_string(
        index=False
    )

    evidence = f"""
Asset ID: {selected_row['UDI']}
Model risk score: {risk_score:.1f}%
Model review threshold: {threshold_percent:.1f}%
Status: {status}

Telemetry statistical context:
{telemetry_text}

Important boundaries:
- Synthetic AI4I industrial predictive-maintenance data.
- Statistical reference ranges are not engineering safety limits.
- Final action belongs to authorised reliability / maintenance staff.
"""

    with st.status(
        "Preparing reliability brief...",
        expanded=True,
    ) as progress:

        st.write(
            "1. Quantitative model evidence prepared."
        )

        final_state = app_engine.invoke(
            {
                "evidence": evidence,
                "technical_explanation": "",
                "handover_note": "",
            }
        )

        st.write(
            "2. LLM translated the evidence into operational language."
        )

        st.write(
            "3. Draft handover note prepared."
        )

        progress.update(
            label="Reliability brief prepared.",
            state="complete",
            expanded=False,
        )

    st.markdown(
        "### Reliability Explanation"
    )

    st.info(
        final_state[
            "technical_explanation"
        ]
    )

    st.markdown(
        "### Draft Shift-Handover Note"
    )

    st.success(
        final_state[
            "handover_note"
        ]
    )


# ============================================================
# 9. TECHNICAL DETAIL — PROGRESSIVE DISCLOSURE
# ============================================================

st.divider()

with st.expander(
    "Technical detail — model validation"
):
    st.write(
        "This section is intended for technical review, not for the "
        "front-line operational workflow."
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Test recall",
        f"{metrics['recall']:.1%}",
    )

    m2.metric(
        "Test precision",
        f"{metrics['precision']:.1%}",
    )

    m3.metric(
        "F2 score",
        f"{metrics['f2']:.2f}",
    )

    m4.metric(
        "Average precision",
        f"{metrics['average_precision']:.2f}",
    )

    st.write(
        {
            "True positives":
                metrics["true_positive"],
            "False positives":
                metrics["false_positive"],
            "False negatives":
                metrics["false_negative"],
            "True negatives":
                metrics["true_negative"],
        }
    )

    st.caption(
        "The prototype uses F2 to give more weight to recall when "
        "selecting the alert threshold. In a production mining use "
        "case, the threshold would be jointly calibrated with "
        "reliability engineers using real failure costs and site risk."
    )


with st.expander(
    "Technical detail — telemetry exploration"
):
    st.write(
        "The original prototype used a 3D telemetry view. "
        "It is retained here only as an optional analytical view so "
        "it does not add cognitive load to the main operational screen."
    )

    try:
        import plotly.express as px

        fig = px.scatter_3d(
            demo_df,
            x="Tool_Wear_min",
            y="Torque_Nm",
            z="Air_Temp_K",
            color="Model_Risk_Score",
            hover_data=[
                "UDI",
                "Process_Temp_K",
                "Rotational_Speed_rpm",
                TARGET,
                "Status",
            ],
            labels={
                "Tool_Wear_min":
                    "Tool Wear (min)",
                "Torque_Nm":
                    "Torque (Nm)",
                "Air_Temp_K":
                    "Air Temperature (K)",
                "Model_Risk_Score":
                    "Model risk score (%)",
            },
            title=(
                "Optional analytical view — "
                "model risk across visible telemetry dimensions"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "This visualisation is exploratory only. The model also "
            "uses process temperature, rotational speed and product "
            "type, so the 3D chart is not the complete decision logic."
        )

    except Exception as exc:
        st.write(
            "Technical visualisation unavailable."
        )


with st.expander(
    "Production-readiness considerations"
):
    st.markdown(
        """
Before production use, I would:

- replace synthetic data with site-specific historical telemetry;
- validate timestamps, sensor quality, asset identifiers and operating context;
- retrain and test against known maintenance and failure outcomes;
- calibrate the alert threshold with reliability engineers;
- compare the cost of false negatives against false positives;
- monitor drift and model performance over time;
- establish access controls, auditability and operational ownership;
- use OEM / engineering limits where applicable rather than statistical reference ranges;
- integrate with SAP / Maximo / Power BI only after workflow and security validation;
- pilot the solution before wider operational deployment.
        """
    )
