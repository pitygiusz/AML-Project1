from pathlib import Path
from scipy.io import arff
import pandas as pd

# https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_100_1000&id=1485

'''
---ABSTRACT---
MADELON is an artificial dataset, which was part of the NIPS 2003 feature selection challenge. This is a two-class classification 
problem with continuous input variables. The difficulty is that the problem is multivariate and highly non-linear.

---DATASET INFORMATION---
MADELON is an artificial dataset containing data points grouped in 32 clusters placed on the vertices of a five-dimensional hypercube 
and randomly labeled 0 or 1. The five dimensions constitute 5 informative features. 15 linear combinations of those features were added 
to form a set of 20 (redundant) informative features. Based on those 20 features one must separate the examples into the 2 classes. 
It was added a number of distractor feature called 'probes' having no predictive power. The order of the features and patterns were randomized.

There is no attribute information provided to avoid biasing the feature selection process.
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

    # Convert target column to binary labels
    if "Class" not in df.columns:
        raise KeyError("Column 'Class' not found in the ARFF file.")

    df["Class"] = (df["Class"] == "2").astype(int)

    # Rename target column
    df = df.rename(columns={"Class": "Y"})

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save deterministically
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    INPUT_FILE = Path("data") / "madelon" / "madelon.arff"
    OUTPUT_FILE = Path("data") / "madelon" / "madelon.csv"

    convert_arff_to_csv(INPUT_FILE, OUTPUT_FILE)
    print(f"Saved CSV to: {OUTPUT_FILE.resolve()}")

