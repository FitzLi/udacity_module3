import sys
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df.columns = sum(df.columns.str.replace('-', '_').str.split().tolist(), [])
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.replace('-', '_').replace(' ', '').lower())
    output_file = sys.argv[1].split('.')[0] + "_cleaned.csv"
    df.to_csv(output_file, index=False)

