import pandas as pd
from pathlib import Path

COLNAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# RUN ONCE

def prepare_car_data(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path, header=None)
    df.columns = COLNAMES

    # Binary target: 1 = acceptable, 0 = unacceptable (car)
    df["Y"] = (df["class"] != "unacc").astype(int)

    # Features without target and original class column
    X = df.drop(columns=["class", "Y"])
    y = df["Y"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # Merge back
    df_processed = pd.concat([X, y], axis=1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed file
    df_processed.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    INPUT_FILE = Path("data") / "car_eval" / "car_eval_raw.csv"
    OUTPUT_FILE = Path("data") / "car_eval" / "car_eval_processed.csv"

    prepare_car_data(INPUT_FILE, OUTPUT_FILE)
    print(f"Saved CSV to: {OUTPUT_FILE.resolve()}")