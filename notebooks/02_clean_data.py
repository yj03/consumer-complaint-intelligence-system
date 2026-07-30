import os
import pandas as pd

print("Program started")

folder_path = r"C:\Users\User\OneDrive\DS_Project\ComplaintsIntelligent\data"
file_path = os.path.join(folder_path, "complaints_sample.csv")

print("Loading sample dataset...")

df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nMissing values:")

missing_values = df.isnull().sum()
missing_percentage = df.isnull().mean() * 100

missing_table = pd.DataFrame({
    "Missing values": missing_values,
    "Missing percentage": missing_percentage
})

missing_table = missing_table.sort_values(
    by="Missing percentage",
    ascending=False
)

print(missing_table.to_string())

# Columns needed for the first NLP model
columns_to_keep = [
    "Complaint ID",
    "Date received",
    "Product",
    "Issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Timely response?"
]

clean_df = df[columns_to_keep].copy()

# Remove rows without complaint narratives
clean_df = clean_df.dropna(
    subset=[
        "Consumer complaint narrative",
        "Product"
    ]
)

# Remove duplicate complaints
clean_df = clean_df.drop_duplicates(
    subset=["Complaint ID"]
)

# Convert the date column
clean_df["Date received"] = pd.to_datetime(
    clean_df["Date received"],
    errors="coerce"
)

# Remove very short narratives
clean_df["Narrative word count"] = (
    clean_df["Consumer complaint narrative"]
    .astype(str)
    .str.split()
    .str.len()
)

clean_df = clean_df[
    clean_df["Narrative word count"] >= 10
]

print("\nClean NLP dataset created")
print("Rows:", clean_df.shape[0])
print("Columns:", clean_df.shape[1])

print("\nProduct distribution:")
print(clean_df["Product"].value_counts().head(10))

print("\nTimely response distribution:")
print(clean_df["Timely response?"].value_counts())

clean_file_path = os.path.join(
    folder_path,
    "complaints_nlp_clean.csv"
)

clean_df.to_csv(
    clean_file_path,
    index=False
)

print("\nClean dataset saved to:")
print(clean_file_path)