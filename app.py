import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
import numpy as np
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Initialize Intelligence Layer
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 

class AgentState(TypedDict):
    sensor_data: str
    diagnostic_report: str
    final_memo: str

# 2. Define Multi-Agent Protocol
def telemetry_ingestion(state: AgentState):
    return {"sensor_data": state["sensor_data"]}

def reliability_analysis(state: AgentState):
    prompt = f"""
    EXECUTIVE DIAGNOSTIC PROTOCOL:
    Analyze the following telemetry to identify operational risk.
    STRUCTURE:
    1. PRIMARY FAILURE MODE: (e.g., Thermal Dissipation)
    2. OPERATIONAL SCENARIO: (Result of 4+ hour continued operation)
    3. FINANCIAL EXPOSURE: (Repair cost vs. preventative isolation)
    DATA: {state['sensor_data']}
    """
    response = llm.invoke(prompt)
    return {"diagnostic_report": response.content}

def operations_communication(state: AgentState):
    prompt = f"""
    OPERATIONAL DIRECTIVE PROTOCOL:
    Convert diagnostic data into a formal shift handover instruction.
    STRUCTURE:
    - ACTION RECOMMENDATION: (e.g., Immediate Isolation)
    - STRATEGIC RATIONALE: (Risk mitigation vs. CapEx protection)
    - SAFETY PROTOCOL: (Specific isolation boundaries)
    DIAGNOSTIC: {state['diagnostic_report']}
    """
    response = llm.invoke(prompt)
    return {"final_memo": response.content}

# 3. Compile Orchestration Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", telemetry_ingestion)
workflow.add_node("engineer", reliability_analysis)
workflow.add_node("coach", operations_communication)
workflow.add_edge(START, "router")
workflow.add_edge("router", "engineer")
workflow.add_edge("engineer", "coach")
workflow.add_edge("coach", END)
app_engine = workflow.compile()

# ==========================================
# 4. QUANTITATIVE RISK MODELING
# ==========================================
@st.cache_data
def load_and_score_data():
    df = pd.read_csv("mining_sensor_stream.csv")
    
    # Statistical Thresholds (90th Percentile) for the Bounding Box
    wear_threshold = df['Tool_Wear_min'].quantile(0.9)
    torque_threshold = df['Torque_Nm'].quantile(0.9)
    temp_threshold = df['Air_Temp_K'].quantile(0.9)
    
    # Standardized Risk Scoring
    df['Risk_Score'] = (
        (df['Tool_Wear_min'] / df['Tool_Wear_min'].max() * 40) +
        (df['Torque_Nm'] / df['Torque_Nm'].max() * 40) +
        ((df['Air_Temp_K'] - df['Air_Temp_K'].min()) / (df['Air_Temp_K'].max() - df['Air_Temp_K'].min()) * 20)
    ).round(1)
    
    def classify_status(row):
        if row['Machine_Failure'] == 1:
            return "Historical Failure"
        if row['Risk_Score'] >= 70:
            return "At Risk"
        return "Operational"
    
    df['Status'] = df.apply(classify_status, axis=1)
    df = df.sort_values(by='Risk_Score', ascending=False).reset_index(drop=True)
    return df, wear_threshold, torque_threshold, temp_threshold

# ==========================================
# 5. EXECUTIVE DASHBOARD (VANTAGE)
# ==========================================
st.set_page_config(page_title="VANTAGE | Operations Intelligence", layout="wide")

st.title("VANTAGE: Operations Intelligence Engine")
st.markdown("Strategic Orchestration Layer | Predictive Maintenance and Capital Protection.")

df, w_thresh, t_thresh, tmp_thresh = load_and_score_data()

# --- SECTION 1: KEY PERFORMANCE INDICATORS ---
st.subheader("Executive Fleet Summary")
col1, col2, col3, col4 = st.columns(4)

at_risk_df = df[df['Status'] == "At Risk"]
exposure_value = len(at_risk_df) * 1.2

col1.metric("Monitored Assets", len(df))
col2.metric("Historical Failures", int(df['Machine_Failure'].sum()))

with col3:
    st.metric("Assets At Risk", len(at_risk_df), delta="Immediate Action Required", delta_color="inverse")

col4.metric("Capital Exposed", f"${exposure_value:.1f}M", delta="Total Seizure Risk", delta_color="inverse")

st.divider()

# --- SECTION 2: RISK FRONTIER ANALYSIS ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("3D Risk Frontier Analysis")
    
    st.info(f"""
    STRATEGIC INSIGHT: The shaded red Danger Zone represents the high-probability failure quadrant. 
    Risk is mathematically confirmed when Tool Wear exceeds {w_thresh:.0f} mins, 
    Torque exceeds {t_thresh:.1f} Nm, and Ambient Temp > {tmp_thresh:.1f} K.
    """)
    
    # --- DANGER ZONE BOUNDING BOX logic ---
    mesh_size = 5 
    x_range = np.linspace(w_thresh, df['Tool_Wear_min'].max(), mesh_size)
    y_range = np.linspace(t_thresh, df['Torque_Nm'].max(), mesh_size)
    z_range = np.linspace(tmp_thresh, df['Air_Temp_K'].max(), mesh_size)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    
    plot_df = df.head(1000)
    
    scatter_fig = px.scatter_3d(plot_df, 
                        x='Tool_Wear_min', y='Torque_Nm', z='Air_Temp_K',
                        color='Status',
                        color_discrete_map={
                            "Historical Failure": "#FF0000", 
                            "At Risk": "#FFD700",           
                            "Operational": "#0000FF"        
                        },
                        opacity=0.7,
                        labels={
                            'Tool_Wear_min': 'Tool Wear (mins)', 
                            'Torque_Nm': 'Torque (Nm)', 
                            'Air_Temp_K': 'Air Temp (K)'
                        })
    
    # THE FIX: Convert scatter_fig.data (tuple) to a list for concatenation
    danger_zone_fig = go.Figure(data=list(scatter_fig.data) + [
        go.Mesh3d(
            x=X, y=Y, z=Z,
            alphahull=0, 
            opacity=0.15, 
            color='#FF4B4B', 
            name='High-Probability Danger Zone',
            showlegend=True
        )
    ])
    
    danger_zone_fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), legend_title_text='Asset Status')
    st.plotly_chart(danger_zone_fig, use_container_width=True)
    
    st.subheader("Asset Drill-down")
    target_list = df['UDI'].tolist()
    selected_udi = st.selectbox("Select Asset Identifier for Scenario Analysis:", target_list[:50])
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    
    display_cols = ["Risk_Score", "Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = ["Risk Score (%)", "Air Temp (K)", "Process Temp (K)", "Rotational Speed (RPM)", "Torque (Nm)", "Tool Wear (mins)"]
    st.dataframe(clean_df, hide_index=True, use_container_width=True)
    
    start_simulation = st.button("Initiate Business Impact Diagnostic", type="primary")

with right_col:
    st.subheader("Multi-Agent Intelligence Log")
    
    if start_simulation:
        with st.status("Executing Intelligence Chain...", expanded=True) as status:
            st.write(f"Informing Agent 1 (Data Ingestion): Processing Asset {selected_udi}...")
            final_state = app_engine.invoke({
                "sensor_data": selected_row.to_string(), 
                "diagnostic_report": "", 
                "final_memo": ""
            })
            st.write("Informing Agent 2 (Reliability Engineering): Analyzing Scenario Impacts...")
            st.write("Informing Agent 3 (Operations Control): Drafting Decision Directive...")
            status.update(label="Analysis Sequence Finalized.", state="complete", expanded=False)
        
        st.markdown("### Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.markdown("### Operational Decision Directive")
        st.success(final_state["final_memo"])