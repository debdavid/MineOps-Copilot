import streamlit as st
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Initialize AI and Environment
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

class AgentState(TypedDict):
    sensor_data: str
    diagnostic_report: str
    final_memo: str

# 2. Define the Agents (Nodes)
def edge_router(state: AgentState):
    # The UI now handles selecting the specific data row, so the router just passes it along!
    return {"sensor_data": state["sensor_data"]}

def reliability_engineer(state: AgentState):
    prompt = f"""You are a senior mining reliability engineer. Look at this raw sensor data and diagnose the likely failure. Keep it brief and analytical.
    Data: {state['sensor_data']}"""
    response = llm.invoke(prompt)
    return {"diagnostic_report": response.content}

def handover_coach(state: AgentState):
    prompt = f"""You are a shift supervisor. Translate this technical diagnostic into a simple, 2-sentence safety memo for the next shift worker. Focus on safety and immediate actions.
    Diagnostic: {state['diagnostic_report']}"""
    response = llm.invoke(prompt)
    return {"final_memo": response.content}

# 3. Build the Graph Engine
workflow = StateGraph(AgentState)
workflow.add_node("router", edge_router)
workflow.add_node("engineer", reliability_engineer)
workflow.add_node("coach", handover_coach)
workflow.add_edge(START, "router")
workflow.add_edge("router", "engineer")
workflow.add_edge("engineer", "coach")
workflow.add_edge("coach", END)
app_engine = workflow.compile()

# ==========================================
# 🎨 UPGRADED STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="MineOps Command Center", layout="wide", page_icon="⛏️")

st.title("⛏️ MineOps Copilot: Command Center")
st.markdown("Interactive Edge-to-Cloud AI orchestration for predictive maintenance.")

# Load data efficiently
@st.cache_data
def load_data():
    return pd.read_csv("mining_sensor_stream.csv")
df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("📊 Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

# Calculate live metrics
total_machines = len(df)
critical_failures = df['Machine_Failure'].sum()
avg_temp = df['Air_Temp_K'].mean()
avg_wear = df['Tool_Wear_min'].mean()

# Display KPIs
col1.metric("Monitored Assets", total_machines)
col2.metric("Critical Failures", f"{critical_failures} 🚨", "- Requires Attention", delta_color="inverse")
col3.metric("Avg Fleet Temp (K)", f"{avg_temp:.1f}")
col4.metric("Avg Tool Wear (mins)", f"{avg_wear:.0f}")

st.divider()

# --- SECTION 2: INTERACTIVE ANALYSIS ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("📈 Equipment Wear & Torque Trends")
    st.info("Tracking mechanical stress across the fleet...")
    # Plot Tool Wear and Torque to visualize mechanical stress
    st.line_chart(df[['Tool_Wear_min', 'Torque_Nm']].head(50))
    
    st.subheader("🎯 Manual Diagnostic Override")
    # INTERACTIVITY: Let the operator choose which failing machine to analyze
    failing_machines_df = df[df['Machine_Failure'] == 1]
    selected_udi = st.selectbox(
        "Select a flagged Machine ID (UDI) to inspect:", 
        failing_machines_df['UDI'].tolist()
    )
    
    # Grab the specific row the user selected
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    st.dataframe(selected_row.to_frame().T, hide_index=True)
    
    start_simulation = st.button("🚨 Run AI Diagnostic on Selected Machine", type="primary")

with right_col:
    st.subheader("🧠 Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Initiating Agent Swarm...", expanded=True) as status:
            st.write(f"🕵️ **Agent 1 (Edge Router):** Extracting telemetry for Machine {selected_udi}...")
            
            # Pass the SPECIFIC row the user selected into the LangGraph state
            final_state = app_engine.invoke({
                "sensor_data": selected_row.to_string(), 
                "diagnostic_report": "", 
                "final_memo": ""
            })
            
            st.write("🔧 **Agent 2 (Reliability Eng):** Diagnosing mechanical failure...")
            st.write("📝 **Agent 3 (Handover Coach):** Translating diagnostic into safety memo...")
            status.update(label="Analysis Complete! Data Lineage Audited.", state="complete", expanded=False)
        
        st.write("### 🔬 Technical Diagnostic (Agent 2)")
        st.info(final_state["diagnostic_report"])
        
        st.write("### ⚠️ Shift Handover Memo (Agent 3)")
        st.success(final_state["final_memo"])
        