# eda.py
# authors: Mavis Wong, Yasmin Hassan and Abeba N. Turi
# date: December 05, 2024

import os
import numpy as np
import pandas as pd
import altair as alt
import click
import matplotlib.pyplot as plt

# Enable the VegaFusion data transformer
alt.data_transformers.enable("vegafusion")

@click.command()
@click.option('--input_csv', type=str, help='Path to input CSV file', required=True)
@click.option('--output_dir', type=str, help='Directory to save visualizations and summary statistics', required=True)
def main(input_csv, output_dir):
    """
    Perform exploratory data analysis (EDA) on the input dataset and save visualizations and statistics.
    """
    # SECTION 1: Load Data
    try:
        train_df = pd.read_csv(input_csv)
        print(f"Data loaded successfully from {input_csv}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # SECTION 2: Data Overview
    print("Training Data Information:")
    print(train_df.info())

    print("\nShape of Training Data:", train_df.shape)

    print("\nSummary Statistics:")
    print(train_df.describe(include="all"))

    # SECTION 3: Data Processing, Feature Engineering and Handling Missing Values
    missing_vals_cols = train_df.columns[train_df.isna().sum() > 0].tolist()
    print("\nColumns with Missing Values:", missing_vals_cols)

    # Define numeric columns explicitly
    numeric_cols = [
        'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
        'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'annual.inc'
    ]

    # Add transformed income-level data
    train_df['annual.inc'] = np.exp(train_df['log.annual.inc'])

    # Loan-to-Income Ratio
    train_df['loan_income_ratio'] = (train_df['installment'] * 12) / train_df['annual.inc']

    # Creating Loan Categories based on FICO score
    loan_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Deep subprime']
    fico_conditions = [
        (train_df['fico'] >= 720),
        (train_df['fico'] < 719) & (train_df['fico'] >= 660),
        (train_df['fico'] < 659) & (train_df['fico'] >= 620),
        (train_df['fico'] < 619) & (train_df['fico'] >= 580),
        (train_df['fico'] < 580)
    ]
    train_df['loan_categories'] = np.select(fico_conditions, loan_categories, default='Unknown')

    # Creating Risk Categories based on FICO score
    conditions = [
        (train_df['fico'] >= 720),
        (train_df['fico'] < 720) & (train_df['fico'] >= 650),
        (train_df['fico'] < 650)
    ]
    categories = ['Low Risk', 'Medium Risk', 'High Risk']
    train_df['risk_category'] = np.select(conditions, categories, default='Unknown')

    # SECTION 4: Visualization (Save to output directory)
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Histograms for Numeric Columns in a Grid
    num_plots = len(numeric_cols)
    n_cols = 3  # Number of columns in the grid
    n_rows = (num_plots // n_cols) + (num_plots % n_cols != 0)  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes to make indexing easier (in case of multi-row grid)
    axes = axes.flatten()

    for i, feat in enumerate(numeric_cols):
        ax = axes[i]
        train_df.groupby("not.fully.paid")[feat].plot.hist(
            bins=40, alpha=0.4, legend=True, density=True, title=f"Histogram of {feat}", ax=ax
        )
        ax.set_xlabel(feat)

    # Hide any unused subplots if the grid is larger than the number of features
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # Adjust layout and save the grid of histograms
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms_grid.png"))
    plt.close(fig)

    # Data distribution of selected features using Altair
    numeric_cols_hists = alt.Chart(train_df).mark_bar().encode(
        alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=20)),  
        y='count()'
    ).properties(
        width=250,
        height=175
    ).repeat(
        ['installment', 'dti'],  
        columns=3
    )
    numeric_cols_hists.save(os.path.join(output_dir, "numeric_feature_distribution.html"))

    # Default Rate by Loan Purpose
    loan_purpose_data = train_df.explode('purpose')
    purpose_risk_chart = alt.Chart(loan_purpose_data).mark_circle().encode(
        x=alt.X('loan_categories:N', title='Loan Categories', sort='-color', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('purpose:N', title='Loan Purpose', sort='color'),
        color=alt.Color('count()', scale=alt.Scale(scheme='viridis'), title='Loan Count'),
        size=alt.Size('count()', title='Loan Count', scale=alt.Scale(range=[50, 1500])),
        tooltip=['purpose', 'loan_categories', 'count()']
    ).properties(
        width=600,
        height=400,
        title="Loan Category vs Loan Purpose"
    )
    purpose_risk_chart.save(os.path.join(output_dir, "loan_category_vs_purpose.html"))

    # Risk Categories Distribution
    categories_hist = alt.Chart(train_df).mark_bar().encode(
        x=alt.X('risk_category:N', title='Risk Categories', axis=alt.Axis(labelAngle=0)),  
        y=alt.Y('count()', title='Count') 
    ).properties(
        height=300,
        width=400,
        title="Distribution of Risk Categories"
    )
    categories_hist.save(os.path.join(output_dir, "risk_categories_distribution.html"))

    # SECTION 5: Correlation Heatmap
    correlation_matrix = train_df[numeric_cols].corr().reset_index().melt('index')
    correlation_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']

    correlation_chart = alt.Chart(correlation_matrix).mark_rect().encode(
        x=alt.X('Variable 1:N', title='', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Variable 2:N', title=''),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable 1', 'Variable 2', 'Correlation']
    ).properties(
        width=400,
        height=400,
        title="Correlation Heatmap"
    )
    correlation_chart.save(os.path.join(output_dir, "correlation_heatmap.html"))

if __name__ == '__main__':
    main()
