import pandas as pd
from pathlib import Path

# https://archive.ics.uci.edu/dataset/320/student+performance

'''
---ABSTRACT---
Predict student performance in secondary education (high school).

---DATASET INFORMATION---
This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, 
demographic, social and school related features and it was collected by using school reports and questionnaires. 
One dataset is provided regarding the performance in Mathematics. Important note: the target attribute G3 has a strong correlation with 
attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 
1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful 
(see paper source for more details). We removed G1 and G2 to avoid data leakage and converted all binary columns to numbers. Also nominal
ones were removed.
'''



# RUN ONLY ONCE

def preprocess_students_data(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path, sep=";")

    
    if "G3" not in df.columns:
        raise KeyError("Column 'G3' not found in the dataset.")

    df["Y"] = (df["G3"] >= 10).astype(int) # if number of final points >=10 then it is a passing grade

   
    cols_to_drop = [col for col in ["G1", "G2", "G3"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    
    binary_mappings = {
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "Y": 1, "N": 0,
        "y": 1, "n": 0,
        "true": 1, "false": 0,
        "True": 1, "False": 0,
        "T": 1, "F": 0,
        "t": 1, "f": 0
    }

    for col in df.columns:
        unique_vals = df[col].dropna().unique()

        # If column has exactly 2 unique values, try to map to 0/1
        if len(unique_vals) == 2:
            # First try common binary strings
            mapped = df[col].map(binary_mappings)

            if mapped.notna().all():
                df[col] = mapped.astype(int)
            else:
                # Fallback: map arbitrary two unique values to 0/1 deterministically
                sorted_vals = sorted(unique_vals, key=lambda x: str(x))
                value_map = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                df[col] = df[col].map(value_map).astype(int)

   
    df = df.select_dtypes(include=["number"])

    if "Y" in df.columns:
        feature_cols = [col for col in df.columns if col != "Y"]
        df = df[feature_cols + ["Y"]]

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    INPUT_FILE = Path("data") / "stud_perf" / "students_raw.csv"
    OUTPUT_FILE = Path("data") / "stud_perf" / "students_processed.csv"

    preprocess_students_data(INPUT_FILE, OUTPUT_FILE)