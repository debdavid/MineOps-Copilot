import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Unlock the safe and wake up the AI
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2) # Low temp = strict/analytical

# ---------------------------------------------------------
# THE CLIPBOARD (Data Lineage)
# ---------------------------------------------------------
class AgentState(TypedDict):
    sensor_data: str         # Where Agent 1 writes the raw data
    diagnostic_report: str   # Where Agent 2 writes the technical diagnosis
    final_memo: str          # Where Agent 3 writes the human-friendly memo

# ---------------------------------------------------------
# THE AGENTS (Nodes)
# ---------------------------------------------------------
def edge_router(state: AgentState):
    print("🕵️ Agent 1 (Edge Router): Scanning sensor stream for anomalies...")
    # Read the simulated data from Phase 2
    df = pd.read_csv("mining_sensor_stream.csv")
    
    # Find the first machine that is failing
    failure_row = df[df["Machine_Failure"] == 1].iloc[0]
    data_string = failure_row.to_string()
    
    print("🚨 Agent 1: Anomaly detected! Passing data to Reliability Engineer.")
    return {"sensor_data": data_string} # Add data to clipboard

def reliability_engineer(state: AgentState):
    print("🔧 Agent 2 (Reliability Eng): Diagnosing the mechanical failure...")
    
    # Send the raw data to the LLM and ask for a diagnosis
    prompt = f"""You are a senior mining reliability engineer. Look at this raw sensor data and diagnose the likely failure. Keep it brief, technical, and analytical.
    Data:
    {state['sensor_data']}"""
    
    response = llm.invoke(prompt)
    return {"diagnostic_report": response.content} # Add diagnosis to clipboard

def handover_coach(state: AgentState):
    print("📝 Agent 3 (Handover Coach): Translating diagnostic into a shift memo...")
    
    # Ask the LLM to translate the technical jargon for the frontline workers
    prompt = f"""You are a shift supervisor. Translate this technical diagnostic into a simple, 2-sentence safety memo for the next shift worker. Focus on safety and what they need to do next. Do not use complex jargon.
    Diagnostic:
    {state['diagnostic_report']}"""
    
    response = llm.invoke(prompt)
    return {"final_memo": response.content} # Add final memo to clipboard

# ---------------------------------------------------------
# THE CONVEYOR BELT (Edges & Graph Architecture)
# ---------------------------------------------------------
print("\n🏗️ Building the Multi-Agent Command Chain...")
workflow = StateGraph(AgentState)

# Add the workers to the factory floor
workflow.add_node("router", edge_router)
workflow.add_node("engineer", reliability_engineer)
workflow.add_node("coach", handover_coach)

# Draw the strict data lineage pathways
workflow.add_edge(START, "router")
workflow.add_edge("router", "engineer")
workflow.add_edge("engineer", "coach")
workflow.add_edge("coach", END)

# Compile the factory
app = workflow.compile()

# ---------------------------------------------------------
# RUN THE SYSTEM
# ---------------------------------------------------------
print("\n🚀 Starting the Multi-Agent System...\n")

# Pass an empty clipboard into the start of the assembly line
final_state = app.invoke({
    "sensor_data": "", 
    "diagnostic_report": "", 
    "final_memo": ""
})

print("\n✅ SYSTEM FINISHED. Here is the final output from the Shift Coach:")
print("=" * 60)
print(final_state["final_memo"])
print("=" * 60)