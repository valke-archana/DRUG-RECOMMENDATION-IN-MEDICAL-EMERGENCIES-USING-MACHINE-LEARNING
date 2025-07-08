import pandas as pd

# Sample Data (Replace this with your actual data)
data = {
    "Symptom": ["muscle_pain", "itching", "altered_sensorium", "dark_urine", "high_fever",
                "mild_fever", "family_history", "nausea", "yellowing_of_eyes", "sweating",
                "unsteadiness", "chest_pain", "fatigue", "abdominal_pain", "joint_pain",
                "diarrhoea", "lack_of_concentration", "red_spots_over_body", "loss_of_appetite", "vomiting"],
    "Importance": [0.019439, 0.016014, 0.016001, 0.015811, 0.015623,
                   0.015100, 0.014786, 0.014670, 0.014079, 0.013823,
                   0.013589, 0.013119, 0.012688, 0.012670, 0.012629,
                   0.012459, 0.011938, 0.011681, 0.011402, 0.011396]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate total sum of Importance
total_importance = df["Importance"].sum()

# Normalize Importance to Percentage
df["Strength (%)"] = (df["Importance"] / total_importance) * 100

# Display DataFrame
print(df)