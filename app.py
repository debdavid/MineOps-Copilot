import streamlit as st
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Initialize AI and Environment
load_dotenv()
# Lowered temperature to 0.1 to force the AI to be highly analytical and formal
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 

class AgentState(TypedDict):
    sensor_data: str
    diagnostic_report: str
    final_memo: str

# 2. Define the Agents (Nodes) - UPGRADED PROMPTS
def telemetry_ingestion(state: AgentState):
    return {"sensor_data": state["sensor_data"]}

def reliability_analysis(state: AgentState):
    prompt = f"""You are a Reliability Analyst for a mining operation. Review this equipment sensor data and provide a clear, business-friendly summary of the mechanical failure. 
    Avoid overly dense engineering jargon. Focus on what component failed, the severity, and the immediate operational impact.
    Data: {state['sensor_data']}"""
    response = llm.invoke(prompt)
    return {"diagnostic_report": response.content}

def operations_communication(state: AgentState):
    prompt = f"""You are a Mine Control Room Supervisor. Write a formal shift handover memo based on the following diagnostic. 
    Output ONLY the memo text. Do not include conversational filler like "Here is the memo" or "Sure". 
    Use professional, authoritative language suitable for heavy industry. Focus on isolation protocols, safety boundaries, and the required maintenance work order.
    Diagnostic: {state['diagnostic_report']}"""
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
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="MineOps Command Center", layout="wide")

st.title("MineOps Copilot: Command Center")
st.markdown("Automated Edge-to-Cloud telemetry analysis and shift handover protocol.")

@st.cache_data
def load_data():
    return pd.read_csv("mining_sensor_stream.csv")
df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

total_machines = len(df)
critical_failures = df['Machine_Failure'].sum()
avg_temp = df['Air_Temp_K'].mean()
avg_wear = df['Tool_Wear_min'].mean()

col1.metric("Monitored Assets", total_machines)
col2.metric("Critical Failures", critical_failures, delta="Attention Required", delta_color="off")
col3.metric("Avg Fleet Temp (K)", f"{avg_temp:.1f}")
col4.metric("Avg Tool Wear (mins)", f"{avg_wear:.0f}")

st.divider()

# --- SECTION 2: INTERACTIVE ANALYSIS ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("Equipment Wear & Torque Trends")
    st.line_chart(df[['Tool_Wear_min', 'Torque_Nm']].head(50))
    
    st.subheader("Manual Diagnostic Override")
    failing_machines_df = df[df['Machine_Failure'] == 1]
    selected_udi = st.selectbox(
        "Select Flagged Machine ID (UDI) for Inspection:", 
        failing_machines_df['UDI'].tolist()
    )
    
    # Isolate the specific row and format it for the business user
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    st.markdown("**Selected Machine Telemetry:**")
    
    display_cols = ["Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = ["Air Temp (K)", "Process Temp (K)", "Speed (RPM)", "Torque (Nm)", "Wear (mins)"]
    
    st.dataframe(clean_df, hide_index=True, use_container_width=True)
    
    start_simulation = st.button("Run AI Diagnostic Protocol", type="primary")

with right_col:
    st.subheader("Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Initiating Analysis Engine...", expanded=True) as status:
            st.write(f"**Telemetry Ingestion Engine:** Extracting payload for Machine {selected_udi}...")
            
            final_state = app_engine.invoke({
                "sensor_data": selected_row.to_string(), 
                "diagnostic_report": "", 
                "final_memo": ""
            })
            
            st.write("**Reliability Analysis Model:** Processing diagnostic...")
            st.write("**Operations Communication Interface:** Generating handover protocol...")
            status.update(label="Analysis Complete. Audited and Logged.", state="complete", expanded=False)
        
        st.write("### Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.write("### Formal Shift Handover Protocol")
        st.success(final_state["final_memo"])
        