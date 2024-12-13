# eda.py
# authors: Mavis Wong, Yasmin Hassan and Abeba N. Turi
# date: December 05, 2024

import os
import numpy as np
import pandas as pd
import altair as alt
import click
import matplotlib.pyplot as plt
import io
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_cleaning import handle_missing_values, add_loan_categories, add_loan_income_ratio, add_risk_categories


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

    # Create the output directory if it does not exist

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # SECTION 2: Data Overview
    
    buffer = io.StringIO()
    train_df.info(buf=buffer)
    info = [row.split() for row in buffer.getvalue().splitlines()[3:-2]]
    info_df = pd.DataFrame(info[2:], columns=info[0]).drop(columns="#")
    info_df.to_csv(os.path.join(output_dir, "tables", "info.csv"))

    # Define numeric columns explicitly
    numeric_cols = [
        'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
        'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'annual.inc'
    ]

    # Add transformed income-level data
    train_df['annual.inc'] = np.exp(train_df['log.annual.inc'])

    # Loan-to-Income Ratio
    train_df = add_loan_income_ratio(train_df, installment_column='installment', income_column='annual.inc')

    # Creating Loan Categories based on FICO score
    train_df = add_loan_categories(train_df, fico_column='fico')


    # Creating Risk Categories based on FICO score
    train_df = add_risk_categories(train_df, fico_column='fico')

    # SECTION 4: Visualization (Save to output directory)

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
            bins=40, alpha=0.4, legend=True, density=True, title=f"{feat}", ax=ax
        )
        ax.set_xlabel(feat)

    # Hide any unused subplots if the grid is larger than the number of features
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # Adjust layout and save the grid of histograms
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "histograms_grid.png"))
    plt.close(fig)


    # Default Rate by Loan Purpose
    loan_purpose_data = train_df.explode('purpose')
    purpose_risk_chart = alt.Chart(loan_purpose_data).mark_circle().encode(
        x=alt.X('loan_categories:N', title='Risk Profile Category', sort='-color', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('purpose:N', title='Loan Purpose', sort='color'),
        color=alt.Color('count()', scale=alt.Scale(scheme='viridis'), title='Loan Count'),
        size=alt.Size('count()', title='Loan Count', scale=alt.Scale(range=[50, 1500])),
        tooltip=['purpose', 'loan_categories', 'count()']
    ).properties(
        width=600,
        height=400
    )
    purpose_risk_chart.save(os.path.join(output_dir, "figures", "loan_category_vs_purpose.png"))

    # Risk Categories Distribution
    categories_hist = alt.Chart(train_df).mark_bar().encode(
        x=alt.X('risk_category:N', title='Risk Categories', axis=alt.Axis(labelAngle=0)),  
        y=alt.Y('count()', title='Count') 
    ).properties(
        height=300,
        width=400
    )
    categories_hist.save(os.path.join(output_dir, "figures", "risk_categories_distribution.png"))

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
    correlation_chart.save(os.path.join(output_dir, "figures", "correlation_heatmap.png"))

    #fico by loan purpose
    purpose_fico_boxplot = alt.Chart(loan_purpose_data).mark_boxplot().encode(
        y=alt.Y('purpose:N', title='Loan Purpose', sort='-x'),  
        x=alt.X('fico:Q', title='FICO Score', scale=alt.Scale(domain=[600, 850]),),  
        color=alt.Color('purpose:N', legend=None),  
        tooltip=['purpose', 'fico']
    ).properties(
        width=400,
        height=200
    )

    purpose_fico_boxplot.save(os.path.join(output_dir, "figures", "boxplot_purpose.png"))

    #Debt to income ratio by risk level
    risk_dti_boxplot = alt.Chart(train_df).mark_boxplot().encode(
        y=alt.Y('risk_category:N', title='Risk Level', sort='-x'),  
        x=alt.X('dti:Q', title='DTI (Debt-to-Income) %', scale=alt.Scale(domain=[0, 35])),  
        color=alt.Color('risk_category:N', legend=None),  
        tooltip=['risk_category', 'dti']
    ).properties(
        width=400,
        height=200
    )


    risk_dti_boxplot.save(os.path.join(output_dir, "figures", "boxplot_risk.png"))


    # SECTION 6: Boxplots (FICO by Loan Purpose and DTI by Risk Category)
    # FICO by loan purpose
    purpose_fico_boxplot = alt.Chart(loan_purpose_data).mark_boxplot().encode(
        y=alt.Y('purpose:N', title='Loan Purpose', sort='-x'),  
        x=alt.X('fico:Q', title='FICO Score', scale=alt.Scale(domain=[600, 850])),  
        color=alt.Color('purpose:N', legend=None),  
        tooltip=['purpose', 'fico']
    ).properties(
        width=400,
        height=200,
        title='Boxplot of FICO Scores by Loan Purpose'
    )

    # Debt to income ratio by risk level
    risk_dti_boxplot = alt.Chart(train_df).mark_boxplot().encode(
        y=alt.Y('risk_category:N', title='Risk Category', sort='-x'),  
        x=alt.X('dti:Q', title='DTI (Debt-to-Income)', scale=alt.Scale(domain=[0, 35])),  
        color=alt.Color('risk_category:N', legend=None),  
        tooltip=['risk_category', 'dti']
    ).properties(
        width=400,
        height=200,
        title='Boxplot of DTI by Risk Category'
    )

if __name__ == '__main__':
    main()
