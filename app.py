import streamlit as st
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Initialize AI and Environment
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# 2. Define the Agent State (Data Lineage)
class AgentState(TypedDict):
    sensor_data: str
    diagnostic_report: str
    final_memo: str

# 3. Define the Agents (Nodes)
def edge_router(state: AgentState):
    df = pd.read_csv("mining_sensor_stream.csv")
    failure_row = df[df["Machine_Failure"] == 1].iloc[0]
    return {"sensor_data": failure_row.to_string()}

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

# 4. Build the Graph Engine
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
# 🎨 STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="MineOps Command Center", layout="wide", page_icon="⛏️")

st.title("⛏️ MineOps Copilot: Multi-Agent Command Center")
st.markdown("A demonstration of Edge-to-Cloud AI orchestration for predictive maintenance and shift safety.")

# Create two columns for the dashboard
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Edge Telemetry Stream")
    st.info("Simulating incoming IoT sensor data from Conveyor Belt Motors...")
    
    # Show a snippet of the normal data
    df_display = pd.read_csv("mining_sensor_stream.csv").head(5)
    st.dataframe(df_display, use_container_width=True)
    
    # The big button to trigger the AI
    start_simulation = st.button("🚨 Run Anomaly Detection Pipeline", type="primary")

with col2:
    st.subheader("🧠 Multi-Agent Analysis Log")
    
    if start_simulation:
        # Show a loading spinner while the agents work
        with st.status("Initiating Agent Swarm...", expanded=True) as status:
            
            st.write("🕵️ **Agent 1 (Edge Router):** Scanning sensor stream...")
            # Run the LangGraph engine!
            final_state = app_engine.invoke({"sensor_data": "", "diagnostic_report": "", "final_memo": ""})
            
            st.write("🔧 **Agent 2 (Reliability Eng):** Diagnosing mechanical failure...")
            st.write("📝 **Agent 3 (Handover Coach):** Translating diagnostic into safety memo...")
            status.update(label="Analysis Complete! Data Lineage Audited.", state="complete", expanded=False)
        
        # Display the Agent outputs in nice UI boxes
        st.write("### 🔬 Technical Diagnostic (Agent 2)")
        st.info(final_state["diagnostic_report"])
        
        st.write("### ⚠️ Shift Handover Memo (Agent 3)")
        st.success(final_state["final_memo"])