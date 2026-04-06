from pathlib import Path
from scipy.io import arff
import pandas as pd

# https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_100_1000&id=40536

'''
---DATA DESCRIPTION---
This data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have 
a four-minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if 
they would like to see their date again. They were also asked to rate their date on six attributes: Attractiveness, Sincerity, 
Intelligence, Fun, Ambition, and Shared Interests. The dataset also includes questionnaire data gathered from participants at 
different points in the process. These fields include: demographics, dating habits, self-perception across key attributes, beliefs on 
what others find valuable in a mate, and lifestyle information. 

Goal: match or mismatch (1 or 0)

'''



def convert_arff_to_csv(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)

    # Decode byte strings to normal Python strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    # Redundant discretized/bin columns
    cols_to_drop = [col for col in df.columns if col.startswith("d_")]

    # Leakage columns
    leakage_cols = ["decision", "decision_o"]

    # Remove unnecessary columns safely
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.drop(columns=leakage_cols, errors="ignore")

    # Rename target
    df = df.rename(columns={"match": "Y"})

    # Remove helper column if present
    if "has_null" in df.columns:
        df = df.drop(columns=["has_null"])

    # Remove rows without target
    df = df.dropna(subset=["Y"])


    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"])
    df["Y"] = df["Y"].astype(int)

    # Split features and target BEFORE one-hot encoding
    X = df.drop(columns=["Y"])
    y = df["Y"]

    # One-hot encode only features
    X = pd.get_dummies(X, drop_first=True, dtype= int)

    # Fill missing numeric values with median
    X = X.fillna(X.median(numeric_only=True))

    # Remove any remaining rows with NaN
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Merge back
    df_processed = pd.concat([X, y], axis=1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save deterministically
    df_processed.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    INPUT_FILE = Path("data") / "speed_dating" / "speed_dating.arff"
    OUTPUT_FILE = Path("data") / "speed_dating" / "speed_dating.csv"

    convert_arff_to_csv(INPUT_FILE, OUTPUT_FILE)
    print(f"Saved CSV to: {OUTPUT_FILE.resolve()}")