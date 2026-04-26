import pandas as pd

# Load your dataset
df = pd.read_csv("herbal_formulation.csv")

def detect_dosha(text):
    text = str(text).lower()

    vata_keywords = ["pain", "anxiety", "dry", "gas", "bloating", "insomnia", "weakness"]
    pitta_keywords = ["burning", "fever", "inflammation", "acidity", "anger", "heat"]
    kapha_keywords = ["cough", "mucus", "heaviness", "lethargy", "cold", "congestion"]

    scores = {"Vata": 0, "Pitta": 0, "Kapha": 0}

    for word in vata_keywords:
        if word in text:
            scores["Vata"] += 1

    for word in pitta_keywords:
        if word in text:
            scores["Pitta"] += 1

    for word in kapha_keywords:
        if word in text:
            scores["Kapha"] += 1

    max_score = max(scores.values())

    if max_score == 0:
        return "Tridosha"

    dominant = [k for k, v in scores.items() if v == max_score]
    return ", ".join(dominant)


# Combine symptoms + description
df["combined_text"] = df["symptoms"] + " " + df["description"]

# Apply dosha detection
df["dosha"] = df["combined_text"].apply(detect_dosha)

# Drop helper column
df.drop(columns=["combined_text"], inplace=True)

# Save new file
df.to_csv("herbal_formulation_with_dosha.csv", index=False)

print("✅ Dosha column added successfully!")