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
# 4. DYNAMIC RISK LOGIC & SORTING
# ==========================================
@st.cache_data
def load_and_score_data():
    df = pd.read_csv("mining_sensor_stream.csv")
    
    # Define Thresholds (Axioms) based on top 10% of global fleet
    w_thresh = df['Tool_Wear_min'].quantile(0.9)
    t_thresh = df['Torque_Nm'].quantile(0.9)
    
    # Calculate Risk Score (Tethering physics to probability)
    df['Risk_Score'] = (
        (df['Tool_Wear_min'] / df['Tool_Wear_min'].max() * 40) +
        (df['Torque_Nm'] / df['Torque_Nm'].max() * 40) +
        ((df['Air_Temp_K'] - df['Air_Temp_K'].min()) / (df['Air_Temp_K'].max() - df['Air_Temp_K'].min()) * 20)
    ).round(1)
    
    def classify(row):
        if row['Machine_Failure'] == 1: return "🔴 Historical Failure"
        if row['Tool_Wear_min'] >= w_thresh or row['Torque_Nm'] >= t_thresh:
            return "🟡 AT RISK (Immediate Action Required)"
        return "🔵 Operational"
    
    df['Status'] = df.apply(classify, axis=1)
    
    # THE CRITICAL FIX: Sort by Risk Score so 'Yellow' machines appear first
    df = df.sort_values(by='Risk_Score', ascending=False).reset_index(drop=True)
    
    return df, w_thresh, t_thresh

# ==========================================
# 5. STREAMLIT UI (Attention System)
# ==========================================
st.set_page_config(page_title="VANTAGE | Ops Intelligence", layout="wide")

st.title("VANTAGE: Operations Intelligence Engine")
st.markdown("Predictive Orchestration: Moving from Lagging Facts to Leading Actions.")

df, w_thresh, t_thresh = load_and_score_data()

# --- SECTION 1: TOP KPI DASHBOARD ---
st.subheader("Live Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

at_risk_df = df[df['Status'] == "🟡 AT RISK (Immediate Action Required)"]

col1.metric("Monitored Assets", len(df))
col2.metric("Historical Failures", df['Machine_Failure'].sum(), help="Lagging Indicators.")

with col3:
    st.metric("Assets At Risk", len(at_risk_df), help="Leading Indicators.")
    if len(at_risk_df) > 0:
        st.markdown(":red[**⚠️ ACTION REQUIRED**]")
    else:
        st.markdown(":green[**✅ SYSTEM OPERATIONAL**]")

col4.metric("Capital Exposed", f"${len(at_risk_df)*1.2:.1f}M", delta="Total Seizure Risk", delta_color="inverse")

st.divider()

# --- SECTION 2: INTERACTIVE SCENARIO PLANNING ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("3.D. Risk Frontier Analysis")
    
    # USER-FRIENDLY INSIGHT
    st.info(f"""
    **Intelligence Insight:** Yellow clusters identify machines entering the **Danger Zone**. 
    Risk triggers when Tool Wear exceeds **{w_thresh:.0f} mins** or Torque exceeds **{t_thresh:.1f} Nm**. 
    Convergence in the top-right corner indicates high risk of immediate **Engine Seizure**.
    """)
    
    # 3D Graph (Using Head of Sorted DF to ensure 'Yellow' dots are included)
    plot_df = df.head(1000)
    fig = px.scatter_3d(plot_df, 
                        x='Tool_Wear_min', y='Torque_Nm', z='Air_Temp_K',
                        color='Status', opacity=0.7,
                        color_discrete_map={
                            "🔴 Historical Failure": "red", 
                            "🟡 AT RISK (Immediate Action Required)": "yellow", 
                            "🔵 Operational": "blue"
                        },
                        labels={
                            'Tool_Wear_min': 'Tool Wear (mins)', 
                            'Torque_Nm': 'Torque (Nm)', 
                            'Air_Temp_K': 'Air Temp (K)'
                        })
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), legend_title_text='Asset Status')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Scenario Override: Machine Inspection")
    # Selection automatically prioritizes the highest-risk machines
    target_list = df['UDI'].tolist() 
    selected_udi = st.selectbox("Select Machine UDI for Analysis (Sorted by Risk):", target_list[:50])
    
    selected_row = df[df['UDI'] == selected_udi].iloc[0]
    
    # USER-FRIENDLY TELEMETRY
    st.write(f"**VANTAGE Health Profile: Machine {selected_udi}**")
    display_cols = ["Risk_Score", "Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min"]
    clean_df = selected_row[display_cols].to_frame().T
    clean_df.columns = ["Risk Score (%)", "Air Temp (K)", "Process Temp (K)", "Rotational Speed (RPM)", "Torque (Nm)", "Tool Wear (mins)"]
    st.dataframe(clean_df, hide_index=True, use_container_width=True)
    
    start_simulation = st.button("Run Business Impact Diagnostic", type="primary")

with right_col:
    st.subheader("Multi-Agent Analysis Log")
    
    if start_simulation:
        with st.status("Orchestrating Agent Chain...", expanded=True) as status:
            st.write(f"**Agent 1 (Ingestion):** Analyzing Machine {selected_udi}...")
            
            final_state = app_engine.invoke({
                "sensor_data": selected_row.to_string(), 
                "diagnostic_report": "", 
                "final_memo": ""
            })
            
            st.write("**Agent 2 (Reliability):** Running Scenario Planning & Impact Analysis...")
            st.write("**Agent 3 (Supervisor):** Finalizing Decision Directive...")
            status.update(label="Analysis Complete.", state="complete", expanded=False)
        
        st.write("### Business Impact Diagnostic")
        st.info(final_state["diagnostic_report"])
        
        st.write("### Formal Decision Directive")
        st.success(final_state["final_memo"])