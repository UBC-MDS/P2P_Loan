# Import 
import os
import pandas as pd
import pandera as pa
import click
import math
from sklearn.model_selection import train_test_split

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

    p2ploan_df = pd.read_csv(data_from)   

    # Ensure directory exists
    if not os.path.isdir(data_to):
        os.makedirs(data_to)

    # Data Validation
    check_prop = pa.Check(lambda s: s.isna().mean() <= 0.05,
                    element_wise=False,
                    error="Too many null values in 'credit.policy' column.")
    schema = pa.DataFrameSchema(
    {
    "credit.policy": pa.Column(int, 
                                checks=[check_prop,
                                        pa.Check.isin([0, 1])], 
                                nullable=True),
    "purpose": pa.Column(
        str, 
        checks=[check_prop,
                pa.Check.isin([
                    "debt_consolidation", 
                    "all_other", 
                    "credit_card", 
                    "home_improvement", 
                    "small_business", 
                    "major_purchase", 
                    "educational"
        ])],
        nullable=True),
    "int.rate": pa.Column(float, checks=[check_prop,pa.Check.in_range(0, 1)], nullable=True),
    "installment": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "log.annual.inc": pa.Column(float, checks=[check_prop,pa.Check.ge(1)], nullable=True),
    "dti": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "fico": pa.Column(int, checks=[check_prop,pa.Check.in_range(300, 900)], nullable=True),
    "days.with.cr.line": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "revol.bal": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "revol.util": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "inq.last.6mths": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "delinq.2yrs": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "pub.rec": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "not.fully.paid": pa.Column(int, checks=[check_prop,pa.Check.isin([0, 1])], nullable=True),
    },
    checks = [
    pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
    pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
    ]

    )

    schema.validate(p2ploan_df)

    train_df, test_df = train_test_split(p2ploan_df, test_size=0.2, random_state=522)

    # Save train data and test data to csv
    train_df.to_csv(os.path.join(data_to, "loan_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "loan_test.csv"), index=False)
    print(f"Train/Test Data successfully saved to {data_to}")

    # Data Validation: Anomalous Correlations
    train_corr = train_df.corr(numeric_only=True)
    neg = train_corr.apply(
        pd.Series.between, axis=1, left=-1, right=-0.9, inclusive="neither"
    ).any().any()

    pos = train_corr.apply(
        pd.Series.between, axis=1, left=0.9, right=1, inclusive="neither"
    ).any().any()


    # Data Validation: Target Distribution

    train_dist = train_df["not.fully.paid"].value_counts(normalize=True)[0]
    test_dist = test_df["not.fully.paid"].value_counts(normalize=True)[0]

    if pos == False and neg == False and math.isclose(train_dist, test_dist, abs_tol=0.05):
        print("All data validation checks pass!")
    elif pos == True or neg == True:
        if math.isclose(train_dist, test_dist, abs_tol=0.05):
            print("Feature/Target correlation exceed maximum acceptable threshold")
        else:
            print("Feature/Target correlation exceed maximum acceptable threshold")
            print("Train/Test Target Distribution Mismatch")
    else:
        print("Train/Test Target Distribution Mismatch")


if __name__ == '__main__':
    main()