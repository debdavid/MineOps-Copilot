import pandas as pd 

print("Fetching the AI4I 2020 Predictive Maintenance Dataset...")

# Download dataset from UCI repo
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# Rename columns to make it easier for the Agents to understand/ read
df.columns = ["UDI", "Product_ID", "Type", "Air_Temp_K", "Process_Temp_K", "Rotational_Speed_rpm", "Torque_Nm", "Tool_Wear_min", 
    "Machine_Failure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

# Simulate a "Live Stream" batch of 100 sensor readings
# Deliberately mix healthy machinery with a few failing machines so the AI has something to diagnose
healthy_machines = df[df["Machine_Failure"] == 0].sample(85, random_state=42)
failing_machines = df[df["Machine_Failure"] == 1].sample(15, random_state=42)

# Combine and shuffle the data to mimic a real-time feed
sensor_stream = pd.concat([healthy_machines, failing_machines]).sample(frac=1).reset_index(drop=True)

# Save this batch locally for our LangGraph agents to pick up later
sensor_stream.to_csv("mining_sensor_stream.csv", index=False)

print("Data downloaded, cleaned, and prepped!")
print(f"Total records in this simulated batch: {len(sensor_stream)}")
print(f"Critical Failures hidden in this batch: {sensor_stream['Machine_Failure'].sum()}")