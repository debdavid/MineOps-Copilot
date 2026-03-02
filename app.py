import streamlit as st
import pandas as pd
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

# 2. Define the Agents (Nodes)
def telemetry_ingestion(state: AgentState):
    return {"sensor_data": state["sensor_data"]}

def reliability_analysis(state: AgentState):
    # This Agent now performs SCENARIO PLANNING for Business Impact
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
    # This Agent ORCHESTRATES the final DECISION Recommendation
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
# STREAMLIT USER INTERFACE (VANTAGE)
# ==========================================
st.set_page_config(page_title="VANTAGE | Ops Intelligence", layout="wide")

st.title("VANTAGE: Operations Intelligence Engine")
st.markdown("Predictive Orchestration Layer: Turning lagging facts into leading actions.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv("mining_sensor_stream.csv")
df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

total_machines = len(df)
historical_failures = df['Machine_Failure'].sum() # LAGGING

# LEADING INDICATOR: We identify "At Risk" machines (high wear + high torque)
at_risk_count = len(df[(df['Tool_Wear_min'] > 180) & (df['Torque_Nm'] > 50) & (df['Machine_Failure'] == 0)])

col1.metric("Monitored Assets", total_machines)
col2.metric("Historical Failures", historical_failures, delta="Lagging Fact", delta_color="off")
col3.metric("Assets At Risk", at_risk_count, delta="Leading Prediction", delta_color="inverse")
col4.metric("Avg Fleet Temp (K)", f"{df['Air_Temp_K'].mean():.1f}")

st.divider()

# --- SECTION 2: INTERACTIVE SCENARIO PLANNING ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("Predictive Stress Trends")
    st.caption("Monitoring the 'Non-Obvious Edges' of Equipment Health")
    
    chart_data = df[['Tool_Wear_min', 'Torque_Nm']].head(50).copy()
    chart_data.columns = ["Tool Wear (Minutes)", "Torque (Newton-meters)"]
    st.line_chart(chart_data)

    st.warning("""
    **VANTAGE Intelligence:** High Tool Wear (Blue) combined with Torque spikes (Light Blue) 
    creates a 'Complexity Curve' leading to engine seizure. Action is required before the fact.
    """)
    
    st.subheader("Scenario Override: Machine Inspection")
    # Show only machines currently at risk but not yet failed
    risk_list = df[(df['Tool_Wear_min'] > 150) & (df['Machine_Failure'] == 0)]['UDI'].tolist()
    
    selected_udi = st.selectbox("Select Machine UDI for 'What-If' Analysis:", risk_list if risk_list else df['UDI'].head(10))
    
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    st.write("**Current Telemetry Payload:**")
    st.dataframe(selected_row[["Air_Temp_K", "Torque_Nm", "Tool_Wear_min"]].to_frame().T, hide_index=True)
    
    start_simulation = st.button("Run Business Impact Diagnostic", type="primary")

with right_col:
    st.subheader("Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Analyzing Risks & Scenario Planning...", expanded=True) as status:
            final_state = app_engine.invoke({"sensor_data": selected_row.to_string(), "diagnostic_report": "", "final_memo": ""})
            status.update(label="Scenario Analysis Complete.", state="complete", expanded=False)
        
        st.write("### Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.write("### Formal Decision Directive")
        st.success(final_state["final_memo"])