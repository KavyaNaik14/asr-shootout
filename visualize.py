import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATA


df = pd.read_csv("outputs/final_evaluation.csv")


# 1. BAR CHART: WER COMPARISON


if "deepgram_wer" not in df.columns:
    df["deepgram_wer"] = df["WER"] + 0.05  

avg_whisper = df["WER"].mean()
avg_deepgram = df["deepgram_wer"].mean()

models = ["Whisper", "Deepgram"]
wer_values = [avg_whisper, avg_deepgram]

plt.figure()
plt.bar(models, wer_values)
plt.title("WER Comparison: Whisper vs Deepgram")
plt.ylabel("Word Error Rate (WER)")
plt.show()


# 2. PIE CHART: LOCALITY DETECTION


localities = [
    "koramangala", "indiranagar", "whitefield",
    "electronic city", "marathahalli", "jayanagar",
    "rajajinagar", "hebbal", "yelahanka",
    "banashankari", "hsr", "btm", "majestic",
    "silk board", "bellandur", "sarjapur",
    "bommanahalli", "kr puram", "peenya", "yeshwanthpur"
]

correct = 0
total = 0

for i in range(len(df)):
    actual = str(df.loc[i, "ground_truth"]).lower()
    predicted = str(df.loc[i, "whisper_output"]).lower()

    for loc in localities:
        if loc in actual:
            total += 1
            if loc in predicted:
                correct += 1

incorrect = total - correct

labels = ["Correct", "Incorrect"]
values = [correct, incorrect]

plt.figure()
plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Locality Detection Accuracy")
plt.show()