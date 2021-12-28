import sys
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df.columns = sum(df.columns.str.split().tolist(), [])
    output_file = sys.argv[1].split('.')[0] + "_cleaned.csv"
    df.to_csv(output_file, index=False)

