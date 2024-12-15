# P2P Loan Risk Analysis Data Analysis Pipeline
# author: Mavis Wong, Yasmin Hassan and Abeba Nigussie Turi
# date: 2024-12-10

.PHONY: all clean

# Default target: run the entire pipeline
all: reports/p2p_lending_risk_analysis_report.html reports/p2p_lending_risk_analysis_report.pdf

# Download data
data/raw/loan_data.csv: scripts/download_data.py
	python scripts/download_data.py \
		--url="https://raw.githubusercontent.com/matmcreative/Lending-Club-Loan-Analysis/refs/heads/main/loan_data.csv" \
		--output_dir=data/raw

# Split data into train/test and validate
data/processed/loan_test.csv data/processed/loan_train.csv: data/raw/loan_data.csv scripts/split_validation.py
	python scripts/split_validation.py \
		--data_from=data/raw/loan_data.csv \
		--data_to=data/processed

# Perform EDA
results/figures/boxplot_purpose.png results/figures/boxplot_risk.png results/figures/correlation_heatmap.png \
results/figures/histograms_grid.png results/figures/loan_category_vs_purpose.png results/figures/risk_categories_distribution.png \
results/tables/info.csv: data/processed/loan_train.csv scripts/eda.py
	python scripts/eda.py \
		--input_csv=data/processed/loan_train.csv \
		--output_dir=results

# Preprocessing
results/models/preprocessor.pickle data/processed/scaled_loan_test.csv data/processed/scaled_loan_train.csv: \
data/processed/loan_test.csv data/processed/loan_train.csv scripts/preprocessing.py
	python scripts/preprocessing.py \
		--data_from=data/processed \
		--data_to=data/processed \
		--preprocessor_to=results/models

# Model Training
results/tables/cv_results.csv results/tables/target_dist.csv results/figures/param_C_tuning.png: \
data/processed/loan_train.csv results/models/preprocessor.pickle scripts/model_training.py
	python scripts/model_training.py \
		--data_from=data/processed \
		--data_to=results/tables \
		--preprocessor_from=results/models/preprocessor.pickle

# Model Tuning
results/models/pipeline.pickle results/tables/model_results.csv: data/processed/loan_train.csv \
results/models/preprocessor.pickle scripts/model_tuning.py
	python scripts/model_tuning.py \
		--data_from=data/processed \
		--data_to=results/tables \
		--preprocessor_from=results/models/preprocessor.pickle \
		--pipeline_to=results/models

# Model Evaluation
results/tables/test_results.csv results/tables/confusion_matrix.csv results/tables/negative_coef.csv \
results/tables/positive_coef.csv: data/processed/loan_train.csv data/processed/loan_test.csv \
results/models/preprocessor.pickle results/models/pipeline.pickle scripts/model_evaluation.py
	python scripts/model_evaluation.py \
		--data_from=data/processed \
		--data_to=results/tables \
		--preprocessor_from=results/models/preprocessor.pickle \
		--pipeline_from=results/models/pipeline.pickle

# Generate reports
reports/p2p_lending_risk_analysis_report.html reports/p2p_lending_risk_analysis_report.pdf: \
reports/p2p_lending_risk_analysis_report.qmd reports/references.bib \
results/figures/* results/tables/*
	quarto render reports/p2p_lending_risk_analysis_report.qmd --to html
	quarto render reports/p2p_lending_risk_analysis_report.qmd --to pdf

# Clean up intermediate and output files
clean:
	rm -rf data/raw/* data/processed/* results/* reports/p2p_lending_risk_analysis_report.*
