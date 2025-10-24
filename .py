import os
import pandas as pd
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG ----------
DATA_FILE = "data/sample_data.csv"   # path to your CSV file
OUTPUT_FILE = "reports/insight_report.txt"
MODEL = "gpt-5"                      # you can also use "gpt-4o" if available
# -----------------------------

# Create folders if missing
os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Initialize OpenAI client
client = OpenAI()

def load_and_clean_data(file_path):
    """Load CSV and do basic cleaning."""
    df = pd.read_csv(file_path)
    original_rows = len(df)
    df = df.drop_duplicates()
    df = df.fillna("N/A")
    cleaned_rows = len(df)
    print(f"Loaded {original_rows} rows → cleaned to {cleaned_rows} rows.")
    return df

def summarize_data(df):
    """Generate a quick data summary."""
    summary = df.describe(include="all").to_string()
    columns = list(df.columns)
    info = f"Columns: {columns}\n\nSummary:\n{summary}"
    return info

def generate_insights(summary_text):
    """Ask GPT to analyze and generate insights."""
    prompt = f"""
You are an expert data analyst agent.
Here is a summary of a dataset:

{summary_text}

Please:
1. Identify 5 key insights, trends, or anomalies.
2. Suggest 2 follow-up analyses that could add value.
3. Keep the response concise and structured.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

def save_report(insights):
    """Save the insights to a report file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"### Auto Data Insight Report\nGenerated: {timestamp}\n\n{insights}\n"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {OUTPUT_FILE}")

def main():
    print("🚀 Running Agentic Data Insight Bot...\n")
    df = load_and_clean_data(DATA_FILE)
    summary_text = summarize_data(df)
    insights = generate_insights(summary_text)
    save_report(insights)
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
