import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from typing import TypedDict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score, confusion_matrix

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

st.set_page_config(page_title="VANTAGE | Operations Intelligence", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

FEATURES = [
    "Type", "Air_Temp_K", "Process_Temp_K",
    "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"
]
NUMERIC_FEATURES = [
    "Air_Temp_K", "Process_Temp_K",
    "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"
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
    "TWF": "TWF", "HDF": "HDF", "PWF": "PWF", "OSF": "OSF", "RNF": "RNF",
}

@st.cache_data
def load_full_dataset():
    for path in ["ai4i2020.csv", "ai4i2020_full.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        df = pd.read_csv(DATA_URL)

    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in FEATURES + [TARGET, "UDI"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

@st.cache_resource
def train_failure_model(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
    ])

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (val_prob >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f2": fbeta_score(y_val, pred, beta=2, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    best = threshold_df.loc[threshold_df["f2"].idxmax()]
    alert_threshold = float(best["threshold"])

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= alert_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    metrics = {
        "alert_threshold": alert_threshold,
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f2": fbeta_score(y_test, test_pred, beta=2, zero_division=0),
        "average_precision": average_precision_score(y_test, test_prob),
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
    }

    test_records = X_test.copy()
    test_records[TARGET] = y_test.values
    test_records["Predicted_Risk"] = test_prob
    test_records["UDI"] = df.loc[test_records.index, "UDI"].values

    return pipeline, metrics, threshold_df, test_records

@st.cache_data
def build_demo_fleet(test_records):
    failures = test_records[test_records[TARGET] == 1]
    healthy = test_records[test_records[TARGET] == 0]
    n_fail = min(15, len(failures))
    n_healthy = 100 - n_fail
    return pd.concat([
        failures.sample(n=n_fail, random_state=42),
        healthy.sample(n=n_healthy, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

load_dotenv()

GROQ_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

def invoke_llm_with_fallback(prompt: str):
    """
    Try supported Groq models in sequence.
    If the generative layer is unavailable, return None so the
    quantitative dashboard still works.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None, "No GROQ_API_KEY configured"

    last_error = None

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            client = ChatGroq(
                model=model_name,
                temperature=0.1,
                groq_api_key=key,
            )
            response = client.invoke(prompt)
            return response.content, model_name
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return None, last_error


class AgentState(TypedDict):
    evidence: str
    technical_explanation: str
    handover_note: str

def evidence_router(state: AgentState):
    return {"evidence": state["evidence"]}

def reliability_explanation(state: AgentState):
    # The quantitative model remains primary; LLM is optional.

    prompt = f"""
You support a reliability engineering team.

BOUNDARIES:
- A quantitative ML model has already produced the risk estimate.
- Do NOT independently diagnose a mechanical failure.
- Do NOT invent a failure probability, safety procedure, isolation boundary, repair cost or engineering limit.
- Explain only the supplied evidence.
- State that a qualified reliability engineer determines maintenance action.

MODEL EVIDENCE:
{state['evidence']}

OUTPUT:
1. RISK SUMMARY
2. CONTRIBUTING SIGNALS
3. NEXT STEP
"""
    content, model_or_error = invoke_llm_with_fallback(prompt)
    if content is not None:
        return {"technical_explanation": content}

    return {"technical_explanation":
        "The optional generative-AI explanation layer is unavailable. "
        "The quantitative model remains operational. A reliability engineer should review the flagged asset. "
        f"Technical note: {model_or_error}"}

def operations_communication(state: AgentState):
    prompt = f"""
Convert this explanation into a concise shift-handover note.

BOUNDARIES:
- Do NOT invent safety procedures.
- Do NOT order isolation unless already authorised in the evidence.
- Do NOT invent financial savings.
- Final maintenance decision belongs to the authorised reliability / maintenance team.

EXPLANATION:
{state['technical_explanation']}

FORMAT:
- ALERT:
- EVIDENCE:
- RECOMMENDED NEXT STEP:
- DECISION OWNER:
"""
    content, _ = invoke_llm_with_fallback(prompt)
    if content is not None:
        return {"handover_note": content}

    return {"handover_note":
        "ALERT: Asset requires reliability review.\n"
        "EVIDENCE: Refer to quantitative risk score and telemetry.\n"
        "RECOMMENDED NEXT STEP: Inspect according to approved site procedures.\n"
        "DECISION OWNER: Authorised reliability / maintenance team."}

workflow = StateGraph(AgentState)
workflow.add_node("evidence", evidence_router)
workflow.add_node("reliability_explainer", reliability_explanation)
workflow.add_node("handover_writer", operations_communication)
workflow.add_edge(START, "evidence")
workflow.add_edge("evidence", "reliability_explainer")
workflow.add_edge("reliability_explainer", "handover_writer")
workflow.add_edge("handover_writer", END)
app_engine = workflow.compile()

df_full = load_full_dataset()
model, metrics, threshold_table, test_records = train_failure_model(df_full)
demo_df = build_demo_fleet(test_records)

demo_df["Risk_Percent"] = (demo_df["Predicted_Risk"] * 100).round(1)
demo_df["Status"] = np.where(
    demo_df["Predicted_Risk"] >= metrics["alert_threshold"],
    "Review Required", "Monitor"
)

st.title("VANTAGE: Operations Intelligence Engine")
st.caption(
    "Predictive-maintenance decision-support prototype | "
    "Synthetic AI4I industrial dataset | Human reliability judgement retained"
)

st.info(
    "Prototype note: this uses a held-out sample from the synthetic AI4I 2020 dataset. "
    "Historical failures are deliberately oversampled in the demo so risk behaviour is visible; "
    "the demo mix is not a real mine fleet failure rate."
)

st.subheader("Fleet Risk Summary")
c1, c2, c3, c4 = st.columns(4)
flagged = demo_df[demo_df["Status"] == "Review Required"]

c1.metric("Demo Assets", len(demo_df))
c2.metric("Historical Failure Examples", int(demo_df[TARGET].sum()))
c3.metric("Assets Flagged for Review", len(flagged))
c4.metric("Model Alert Threshold", f"{metrics['alert_threshold']:.2f}")

st.caption(
    "The threshold is selected on validation data to maximise F2 score, which places more weight on recall. "
    "In production it would be calibrated with reliability engineers using real site data and the cost of missed failures versus false alarms."
)

with st.expander("Model validation"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Recall", f"{metrics['recall']:.1%}")
    m2.metric("Test Precision", f"{metrics['precision']:.1%}")
    m3.metric("F2 Score", f"{metrics['f2']:.2f}")
    m4.metric("Average Precision", f"{metrics['average_precision']:.2f}")
    st.write({
        "True positives": metrics["true_positive"],
        "False positives": metrics["false_positive"],
        "False negatives": metrics["false_negative"],
        "True negatives": metrics["true_negative"],
    })

st.divider()
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("3D Equipment Risk View")
    fig = px.scatter_3d(
        demo_df,
        x="Tool_Wear_min",
        y="Torque_Nm",
        z="Air_Temp_K",
        color="Risk_Percent",
        hover_data=[
            "UDI", "Process_Temp_K", "Rotational_Speed_rpm",
            TARGET, "Status"
        ],
        labels={
            "Tool_Wear_min": "Tool Wear (min)",
            "Torque_Nm": "Torque (Nm)",
            "Air_Temp_K": "Air Temp (K)",
            "Risk_Percent": "Predicted risk (%)",
        },
        title="Model-derived risk across visible telemetry dimensions",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Colour represents model-derived risk. The model also uses process temperature, rotational speed and product type."
    )

with right_col:
    st.subheader("Asset Drill-down")
    selected_udi = st.selectbox("Select Asset Identifier:", demo_df["UDI"].tolist())
    selected_row = demo_df[demo_df["UDI"] == selected_udi].iloc[0]

    st.metric("Predicted Failure Risk", f"{selected_row['Risk_Percent']:.1f}%")
    st.metric("Decision-Support Status", selected_row["Status"])

    display_cols = [
        "Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm",
        "Torque_Nm", "Tool_Wear_min"
    ]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = [
        "Air Temp (K)", "Process Temp (K)", "Rotational Speed (RPM)",
        "Torque (Nm)", "Tool Wear (min)"
    ]
    st.dataframe(clean_df, hide_index=True, use_container_width=True)

    if selected_row[TARGET] == 1:
        st.warning(
            "Evaluation note: this held-out record is labelled as a historical failure in the source dataset. "
            "That label is shown only for evaluation and would not be known for an unseen live asset."
        )

st.divider()
st.subheader("AI-Assisted Reliability Explanation")
st.write(
    "The quantitative model creates the risk signal. "
    "The LLM explains the evidence and drafts a handover message; it does not authorise maintenance action."
)

if st.button("Generate Reliability Review Brief", type="primary"):
    evidence = f"""
Asset ID: {selected_row['UDI']}
Model risk estimate: {selected_row['Risk_Percent']:.1f}%
Alert threshold: {metrics['alert_threshold'] * 100:.1f}%
Status: {selected_row['Status']}

Telemetry:
- Air temperature: {selected_row['Air_Temp_K']} K
- Process temperature: {selected_row['Process_Temp_K']} K
- Rotational speed: {selected_row['Rotational_Speed_rpm']} rpm
- Torque: {selected_row['Torque_Nm']} Nm
- Tool wear: {selected_row['Tool_Wear_min']} min

Dataset note: synthetic AI4I industrial predictive-maintenance data.
"""

    with st.status("Preparing reliability review...", expanded=True) as status:
        st.write("Stage 1 — quantitative model evidence prepared.")
        final_state = app_engine.invoke({
            "evidence": evidence,
            "technical_explanation": "",
            "handover_note": "",
        })
        st.write("Stage 2 — LLM explains the model evidence.")
        st.write("Stage 3 — LLM drafts a bounded handover note.")
        status.update(label="Review brief prepared.", state="complete", expanded=False)

    st.markdown("### Reliability Explanation")
    st.info(final_state["technical_explanation"])
    st.markdown("### Draft Shift-Handover Note")
    st.success(final_state["handover_note"])

with st.expander("Production-readiness considerations"):
    st.markdown("""
- Replace synthetic data with site-specific historical telemetry.
- Validate sensor quality, timestamps and asset context.
- Retrain and test against known maintenance/failure outcomes.
- Set the operational threshold with reliability engineers.
- Monitor model drift and false-positive / false-negative rates.
- Apply identity/access controls and auditability.
- Keep authorised reliability/maintenance staff accountable for consequential action.
- Integrate with SAP/Maximo/Power BI only after workflow and security validation.
- Pilot before operational rollout.
""")
