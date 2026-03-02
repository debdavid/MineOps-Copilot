import streamlit as st
import pandas as pd
import plotly.express as px
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Initialize AI and Environment
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 

class AgentState(TypedDict):
    sensor_data: str
    diagnostic_report: str
    final_memo: str

# 2. Define the Agents (Nodes) - "Snappy" Prompts to reduce Cognitive Load
def telemetry_ingestion(state: AgentState):
    return {"sensor_data": state["sensor_data"]}

def reliability_analysis(state: AgentState):
    # The DIAGNOSER (Intelligence Engine)
    prompt = f"""
    You are a Senior Reliability Engineer. Analyze this telemetry and perform a 
    Risk-Adjusted Diagnostic using this structure:

    1. AT-RISK FAILURE: (e.g., Heat Dissipation, Overstrain)
    2. SCENARIO: (If run for 4+ more hours, what is the 'Catastrophic' result?)
    3. BUSINESS IMPACT: (Estimated repair cost vs. 20-min cooling shutdown)

    Keep it snappy. Focus on saving the '$1M repair' cost.
    Data: {state['sensor_data']}
    """
    response = llm.invoke(prompt)
    return {"diagnostic_report": response.content}

def operations_communication(state: AgentState):
    # The ACTION ORCHESTRATOR (Agent Chain)
    prompt = f"""
    You are a Mine Control Room Supervisor. Convert the diagnostic into a 
    Formal Decision Directive. 

    - RECOMMENDATION: [e.g., SHUT DOWN FOR 20 MINS NOW]
    - THE RATIONALE: [Why this prevents a $1M seizure]
    - PROTOCOL: [Specific safety boundaries for isolation]

    Tone: Authoritative, Professional, Urgent.
    Diagnostic: {state['diagnostic_report']}
    """
    response = llm.invoke(prompt)
    return {"final_memo": response.content}

# 3. Build the Graph Engine
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
# 4. DATA LOGIC & RISK SCORING
# ==========================================
def calculate_risk_score(row):
    """Tethers sensor data to a 0-100 Risk Score."""
    # Weighting: Wear (40%), Torque (40%), Temp (20%)
    wear_score = min((row['Tool_Wear_min'] / 250) * 100, 100)
    torque_score = min((row['Torque_Nm'] / 70) * 100, 100)
    temp_score = min(((row['Air_Temp_K'] - 280) / 40) * 100, 100)
    
    total_score = (wear_score * 0.4) + (torque_score * 0.4) + (temp_score * 0.2)
    return round(total_score, 1)

def classify_risk(row):
    if row['Machine_Failure'] == 1:
        return "🔴 Historical Failure"
    if row['Risk_Score'] > 75:
        return "🟡 AT RISK (Action Required)"
    return "🔵 Operational"

@st.cache_data
def load_data():
    df = pd.read_csv("mining_sensor_stream.csv")
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    df['Status'] = df.apply(classify_risk, axis=1)
    return df

# ==========================================
# 5. STREAMLIT UI (Attention System)
# ==========================================
st.set_page_config(page_title="VANTAGE | Ops Intelligence", layout="wide")

st.title("VANTAGE: Operations Intelligence Engine")
st.markdown("Predictive Orchestration: Moving from Lagging Facts to Leading Actions.")

df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

at_risk_df = df[df['Status'] == "🟡 AT RISK (Action Required)"]

col1.metric("Monitored Assets", len(df))
col2.metric("Historical Failures", df['Machine_Failure'].sum(), help="Lagging Indicators.")

with col3:
    st.metric("Assets At Risk", len(at_risk_df), help="Leading Indicators.")
    if len(at_risk_df) > 0:
        st.markdown(":red[**⚠️ ACTION REQUIRED**]")
    else:
        st.markdown(":green[**✅ SYSTEM OPERATIONAL**]")

col4.metric("Capital Exposed", f"${len(at_risk_df)*1.2:.1f}M", delta="Repair Risk", delta_color="inverse")

st.divider()

# --- SECTION 2: INTERACTIVE RISK FRONTIER ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("3.D. Risk Frontier Analysis")
    
    st.info(f"""
    **Current Analysis:** Yellow clusters identify assets drifting toward historic 'Danger Zones'. 
    Risk is calculated by the convergence of **Wear, Torque, and Ambient Heat**.
    """)
    
    # 3D Scatter Plot
    plot_df = df.head(1000).copy()
    fig = px.scatter_3d(plot_df, 
                        x='Tool_Wear_min', y='Torque_Nm', z='Air_Temp_K',
                        color='Status', 
                        color_discrete_map={
                            "🔴 Historical Failure": "red", 
                            "🟡 AT RISK (Action Required)": "yellow", 
                            "🔵 Operational": "blue"
                        },
                        opacity=0.7,
                        labels={'Tool_Wear_min': 'Wear', 'Torque_Nm': 'Torque', 'Air_Temp_K': 'Air Temp'})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), legend_title_text='Asset Status')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Scenario Override: Machine Inspection")
    # Selection prioritizing at-risk machines
    risk_list = at_risk_df['UDI'].tolist()
    selected_udi = st.selectbox("Select Machine UDI for Analysis:", risk_list if risk_list else df['UDI'].head(10))
    
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    
    # Restored Full Telemetry + Risk Score
    st.write(f"**VANTAGE Health Profile: Machine {selected_udi}**")
    display_cols = ["Risk_Score", "Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = ["Risk Score (%)", "Air Temp (K)", "Process Temp (K)", "Speed (RPM)", "Torque (Nm)", "Wear (mins)"]
    st.dataframe(clean_df, hide_index=True, use_container_width=True)
    
    start_simulation = st.button("Run Business Impact Diagnostic", type="primary")

with right_col:
    st.subheader("Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Analyzing Risks & Scenario Planning...", expanded=True) as status:
            st.write(f"**Agent 1 (Ingestion):** Analyzing Machine {selected_udi}...")
            
            final_state = app_engine.invoke({
                "sensor_data": selected_row.to_string(), 
                "diagnostic_report": "", 
                "final_memo": ""
            })
            
            st.write("**Agent 2 (Reliability):** Running Scenario Planning...")
            st.write("**Agent 3 (Supervisor):** Finalizing Decision Directive...")
            status.update(label="Analysis Complete.", state="complete", expanded=False)
        
        st.write("### 🧠 Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.write("### 🚨 Formal Decision Directive")
        st.success(final_state["final_memo"])
        