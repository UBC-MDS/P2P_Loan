
# Lending Club Loan Data Analysis Project

## Contributors
- Mavis Wong
- Yasmin Hassan
- Abeba Nigussie Turi

## Project Summary
This project focuses on predicting loan defaults by analyzing historical data from the Lending Club platform. By performing detailed data preprocessing, exploratory data analysis (EDA), and feature engineering, the project aims to identify key factors influencing loan repayment outcomes. Advanced deep learning models are employed to detect patterns in borrower behavior, providing a robust framework for understanding repayment risks. The ultimate objective is to deliver actionable insights that enhance borrower risk assessment and support informed decision-making in loan management.


## How to Run the Analysis
1. **Clone the Repository**:
   ```bash
   git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis

2. **Using environment.yml**

This is the recommended method to set up the environment that can allow you run the file
  Create the Conda environment:

    bash
    conda env create -f environment.yaml
    
  Activate the environment:

    bash
    conda activate loan_risk522

## Using the Container Image
To use the containerized environment for this project follow this

1. Pull the Latest Image Pull the latest version of the container image from Docker Hub
(docker pull yasmin2424/p2p_loan_risk_analysis:latest)

2. Run the Container Launch the container to start working on the project
(docker run -it --rm -p 8888:8888 yasmin2424/p2p_loan_risk_analysis:latest)

3. Access your project files if you want to work with your local project files, mount the project directory


## Dependencies
  - numpy
  - pandas
  - python=3.11.6
  - scikit-learn
  - altair
  - matplotlib
  - vegafusion-python-embed=1.6.9
  - vegafusion=1.6.9
  - vl-convert-python[version='>=1.6.0']
  - ipykernel
  - conda-lock

## License
- **Code**: This project uses the MIT License. See [LICENSE.md]for details.
