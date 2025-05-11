import pandas as pd
import json

json_path = r"c:\Users\Asus\python\Project\Document_analyzer\Data\text.data.jsonl"

casedata = []

# Open the JSONL file normally if it's not compressed
with open(json_path, 'r', encoding='utf-8') as in_file:
    for index, line in enumerate(in_file):
        if index > 250000:
            break
        j = json.loads(line)
        opinions = j.get('casebody', {}).get('data', {}).get('opinions', [])
        citations = j.get('citations', [{}])

        for opinion in opinions:
            casedata.append([
                opinion.get('author', 'Unknown'),
                j.get('name_abbreviation', 'N/A'),
                citations[0].get('cite', 'N/A'),
                j.get('decision_date', 'N/A'),
                opinion.get('type', 'N/A'),
                opinion.get('text', 'N/A')
            ])

casedf = pd.DataFrame(casedata, columns=['Author', 'Name', 'Citation', 'DecisionDate', 'OpinionType', 'OpinionText'])

# Save subset of text data
subset_size = len(casedf) // 16  
text_data = "\n".join(casedf["OpinionText"].astype(str)[:subset_size])

with open("opinion_texts_subset.txt", "w", encoding="utf-8") as file:
    file.write(text_data)

print("Subset of text data saved successfully!")

print("preprocessing started")


# Load the dataset (Modify the filename accordingly) # Change to the actual path
df = casedf

# Drop completely empty rows
df.dropna(how="all", inplace=True)

# Fill missing values in important columns
df["Author"] = df["Author"].fillna("Unknown")
df["Name"] = df["Name"].fillna("Unknown")
df["OpinionText"] = df["OpinionText"].fillna("No opinion available")

# Normalize DecisionDate format (Ensure all dates are in YYYY-MM-DD format)
df["DecisionDate"] = pd.to_datetime(df["DecisionDate"], errors="coerce").dt.strftime("%Y-%m-%d")

# Clean OpinionText (remove extra spaces, newlines, and tabs)
df["OpinionText"] = df["OpinionText"].str.replace(r"[\n\t]", " ", regex=True).str.strip()



# Take only 1/32 of the dataset
sampled_df = df.sample(frac=1/32, random_state=42)

# Save sampled data to a text file
output_txt_path = "output_chunks.txt"
with open(output_txt_path, "w", encoding="utf-8") as file:
    for _, row in sampled_df.iterrows():
        file.write(f"Case: {row['Name']}\n")
        file.write(f"Author: {row['Author']}\n")
        file.write(f"Citation: {row['Citation']}\n")
        file.write(f"Date: {row['DecisionDate']}\n")
        file.write(f"Type: {row['OpinionType']}\n")
        file.write(f"Text: {row['OpinionText']}\n")
        file.write("=" * 50 + "\n\n")  # Separator for readability

print(f"✅ Preprocessed and saved 1/32 of the dataset to {output_txt_path}")

