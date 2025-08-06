import pandas as pd
import sys

def convert_jsonl_to_parquet(input_path, output_path):
    # Read JSONL file
    df = pd.read_json(input_path, lines=True)
    
    # Save as Parquet
    df.to_parquet(output_path, index=False)
    print(f"Converted {input_path} to {output_path}")
    print(f"Number of rows: {len(df)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet.py <input_jsonl> <output_parquet>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_jsonl_to_parquet(input_file, output_file)
