# Import 
import os
import pandas as pd
import pandera as pa
import click
import math
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_validation import validate
from src.write_csv import write_csv

@click.command()
@click.option('--data_from', type=str, help="Path to raw data")
@click.option('--data_to', type=str, help="Path to directory where processed data will be written to")
def main(data_from, data_to):
    try:
        p2ploan_df = pd.read_csv(data_from)
        print(f"Data loaded successfully from {data_from}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
  
    # Ensure directory exists
    if not os.path.isdir(data_to):
        os.makedirs(data_to)

    # Data Validation
    validate(p2ploan_df)

    train_df, test_df = train_test_split(p2ploan_df, test_size=0.2, random_state=522)
 
    # Save train data and test data to csv
    write_csv(train_df, data_to, "loan_train.csv", index=False)
    write_csv(test_df, data_to, "loan_test.csv", index=False)
    print(f"Train/Test Data successfully saved to {data_to}")

if __name__ == '__main__':
    main()