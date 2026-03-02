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

# 2. Define the Agents (Nodes)
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
st.markdown("Predictive Orchestration: Moving knowledge work up an abstraction layer.")

@st.cache_data
def load_data():
    return pd.read_csv("mining_sensor_stream.csv")
df = load_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

total_machines = len(df)
historical_failures = df['Machine_Failure'].sum()
at_risk_count = len(df[
    (df['Machine_Failure'] == 0) & 
    ((df['Tool_Wear_min'] > 180) | (df['Torque_Nm'] > 60))
])

col1.metric("Monitored Assets", total_machines)
col2.metric("Historical Failures", historical_failures, help="Lagging Fact.")

# FIX: Action Required is now RED when assets are at risk
with col3:
    st.metric("Assets At Risk", at_risk_count, help="Leading Indicator.")
    if at_risk_count > 0:
        st.markdown(":red[**⚠️ ACTION REQUIRED**]")
    else:
        st.markdown(":green[**✅ SYSTEM OPERATIONAL**]")

# NEW INSIGHT: Financial Protection Metric
protected_value = at_risk_count * 1.2 # Assuming $1.2M average repair cost saved
col4.metric("Capital At Risk", f"${protected_value:.1f}M", delta="Critical Exposure", delta_color="inverse")

st.divider()

# --- SECTION 2: INTERACTIVE SCENARIO PLANNING ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("3.D. Risk Frontier Analysis")
    
    st.info("""
    **Insight:** This 3D space visualizes the convergence of **Wear, Torque, and Heat**. 
    Assets floating in the top-right-back corner have reached the physical 'Frontier' and are 
    highest priority for intervention.
    """)
    
    # NEW INSIGHT: 3D Scatter Plot
    plot_df = df.head(800).copy()
    plot_df['Status'] = plot_df['Machine_Failure'].map({1: 'Historical Failure', 0: 'Operational'})
    
    fig = px.scatter_3d(plot_df, 
                        x='Tool_Wear_min', y='Torque_Nm', z='Air_Temp_K',
                        color='Status', 
                        symbol='Status',
                        opacity=0.7,
                        color_discrete_map={'Historical Failure': 'red', 'Operational': 'blue'},
                        labels={'Tool_Wear_min': 'Wear', 'Torque_Nm': 'Torque', 'Air_Temp_K': 'Air Temp'})
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Scenario Override: Machine Inspection")
    risk_list = df[(df['Tool_Wear_min'] > 150) & (df['Machine_Failure'] == 0)]['UDI'].tolist()
    selected_udi = st.selectbox("Select Machine UDI for 'What-If' Analysis:", risk_list if risk_list else df['UDI'].head(10))
    
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    
    # Restored Full Telemetry
    display_cols = ["Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = ["Air Temp (K)", "Process Temp (K)", "Speed (RPM)", "Torque (Nm)", "Wear (mins)"]
    st.dataframe(clean_df, hide_index=True, use_container_width=True)
    
    start_simulation = st.button("Run Business Impact Diagnostic", type="primary")

with right_col:
    st.subheader("Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Initializing Analysis Engine...", expanded=True) as status:
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