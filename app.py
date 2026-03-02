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

# 2. Define the Agents (Nodes) - "Snappy" Prompts for high-stress decision making
def telemetry_ingestion(state: AgentState):
    return {"sensor_data": state["sensor_data"]}

def reliability_analysis(state: AgentState):
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
st.markdown("Predictive Orchestration Layer: Identifying 'Non-Obvious Edges' before equipment failure.")

@st.cache_data
def load_data():
    return pd.read_csv("mining_sensor_stream.csv")
df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

total_machines = len(df)
historical_failures = df['Machine_Failure'].sum()

# LEADING INDICATOR: Analyzing the 'Complexity Curve'
# Flagging machines hitting historic danger thresholds (Wear > 180 or Torque > 60)
at_risk_count = len(df[
    (df['Machine_Failure'] == 0) & 
    ((df['Tool_Wear_min'] > 180) | (df['Torque_Nm'] > 60))
])

col1.metric("Monitored Assets", total_machines)
col2.metric("Historical Failures", historical_failures, help="Lagging Fact: Total failures already recorded.")
col3.metric("Assets At Risk", at_risk_count, delta="Action Required", help="Leading Indicator: Assets currently trending toward failure.")
col4.metric("Avg Fleet Temp (K)", f"{df['Air_Temp_K'].mean():.1f}")

st.divider()

# --- SECTION 2: INTERACTIVE SCENARIO PLANNING ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("VANTAGE Risk Heatmap")
    
    # UI FIX: Explanation placed ABOVE the chart for clarity
    st.info("""
    **VANTAGE Intelligence:** Red zones identify historic failure clusters. 
    As assets drift into these **Danger Zones** (High Tool Wear + High Torque), 
    the probability of a $1M engine seizure increases exponentially.
    """)
    
    st.caption("Visualizing the 'Non-Obvious Edges' of Heat & Stress")
    
    # Heatmap setup
    heatmap_data = df.head(500)
    fig = px.density_heatmap(heatmap_data, x="Tool_Wear_min", y="Torque_Nm", 
                             z="Machine_Failure", histfunc="sum",
                             labels={'Tool_Wear_min':'Tool Wear (mins)', 'Torque_Nm':'Torque (Nm)'},
                             color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Scenario Override: Machine Inspection")
    # Show machines at risk
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
        
        st.write("### 🧠 Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.write("### 🚨 Formal Decision Directive")
        st.success(final_state["final_memo"])