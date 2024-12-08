
# P2P Online Lending Default Prediction- A Usecase on LendingClub Default Risk

#### Contributors: Mavis Wong, Yasmin Hassan and Abeba Nigussie Turi

## About the Project
This work intends to leaverage machine learning models to predict borrower behaviour and hence probability of default. More specifically, the work focuses aszxon predicting loan defaults using historical data from the Lending Club platform. By applying advanced preprocessing techniques, exploratory data analysis (EDA), and a Logistic Regression model, we uncover patterns and trends in borrower risk profiles. 

The final model demonstrated strong performance on unseen test data, achieving an accuracy of 84.0%. Out of 1,916 test cases, the model correctly predicted 1,608 cases, with 308 incorrect predictions. These errors included both false positives (predicting a loan default when it didn’t occur) and false negatives (failing to predict an actual default).

While false negatives pose a greater risk in financial decision-making, this model provides actionable insights to improve risk management and reduce potential financial losses for the platform. Despite its promising predictive capabilities, further research is needed to enhance the model's accuracy and better understand the characteristics of misclassified loans. Such improvements could play a crucial role in minimizing financial risks and maximizing the model’s effectiveness in peer-to-peer lending platforms.

# Report
You can acces the final report at [here](https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/blob/main/notebooks/p2p_lending_risk_analysis_report.ipynb)



# How to Run the Analysis
1. **Clone the Repository**:
   ```bash
   git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis
   ```
## OPTION 1
**Using environment.yml option**

This is the recommended method to set up the environment that can allow you run the file
  1. Create the Conda environment:
   ```bash
    conda env create -f environment.yml
   ```
  2. Activate the environment:

    ```bash
    conda activate loan_risk522
    ```
 3. Verify the environment setup using this.

`python -c "import pandas as pd; print('Environment set up successfully!')"`

## OPTION 2

### Using the Docker Container Image
To use the containerized environment for this project follow this steps if it is your first time

1. Ensure you have Docker and Docker Compose installed.
2. Clone this repository and navigate to the root directory. [here](git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis)
2. Clone this repository and navigate to the root directory. [here](git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis)
3. Run: 
```bash
   docker-compose up
   ```
4. Access the Jupyter Notebook interface at http://localhost:8888.

5. To run the analysis, open a terminal on jupyterlab and run the following commands:
```bash
python scripts/download_data.py --url "https://raw.githubusercontent.com/matmcreative/Lending-Club-Loan-Analysis/refs/heads/main/loan_data.csv" --output_dir "data/raw"

python scripts/split_validation.py --data_from "data/raw/loan_data.csv" --data_to "data/processed"

python scripts/eda.py --input_csv "data/processed/loan_train.csv" --output_dir "results"

python scripts/preprocessing.py --data_from "data/processed" --data_to "data/processed" --preprocessor_to "results/models"

python scripts/model_training.py --data_from "data/processed" --data_to "results/tables" --preprocessor_from "results/models/preprocessor.pickle"

python scripts/model_tuning.py --data_from "data/processed" --data_to "results/tables" --preprocessor_from "results/models/preprocessor.pickle" --pipeline_to "results/models"

python scripts/model_evaluation.py --data_from "data/processed" --data_to "results/tables" --preprocessor_from "results/models/preprocessor.pickle" --pipeline_from "results/models/pipeline.pickle"
quarto render reports/p2p_lending_risk_analysis_report.qmd --to html
quarto render reports/p2p_lending_risk_analysis_report.qmd --to pdf
```
6. To shut down the container and clean up the resources, type Cntrl + C in the terminal where you launched the container, and then type 
```bash
   docker compose rm
```




## Dependencies
[Docker](https://www.docker.com)

conda (version 23.9.0 or higher)

conda-lock (version 2.5.7 or higher)

mamba (version 1.5.8 or higher)

nb_conda_kernels (version 2.3.1 or higher)

Python and packages listed in [here](https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/blob/main/environment.yml)


## Adding a new dependency and Updating Environment
1. Add the dependency to the environment.yml file on a new branch. If the package is pip installed, it should also be added to Dockerfile with command RUN pip install <package_name> = <version>

2. Run conda-lock -k explicit --file environment.yml -p linux-64 to update the conda-linux-64.lock file.

3. Re-run the scripts above using either options above.

4. If the environment.yml file is updated (e.g., new dependencies are added), you can update your existing environment with:

 ```conda env update -f environment.yaml --prune```


## License
- **Code**:
If you are re-using/re-mixing please provide attribution and link to this webpage. 
 This project uses the MIT License. See the [the license file](LICENSE.md) for details.


## References
1. Cai, S., Lin, X., Xu, D., & Fu, X. (2016). Judging online peer-to-peer lending behavior: A comparison of first-time and repeated borrowing requests. Information & Management, 53(7), 857-867.Consumer
https://www.sciencedirect.com/science/article/abs/pii/S0378720616300805

2. Coşer, A., Maer-Matei, M. M., & Albu, C. (2019). PREDICTIVE MODELS FOR LOAN DEFAULT RISK ASSESSMENT. Economic Computation & Economic Cybernetics Studies & Research, 53(2). https://ecocyb.ase.ro/nr2019_2/9.%20Coser%20Al.%20Crisan%20Albu%20(T).pdf

3. Equifax. (n.d.). *Credit score ranges.* Retrieved November 20, 2024, from [https://www.equifax.com/personal/education/credit/score/articles/-/learn/credit-score-ranges/](https://www.equifax.com/personal/education/credit/score/articles/-/learn/credit-score-ranges/)

4. Financial Protection Bureau. (n.d.). *Borrower risk profiles: Student loans*. Retrieved November 20, 2024, from [https://www.consumerfinance.gov/data-research/consumer-credit-trends/student-loans/borrower-risk-profiles/](https://www.consumerfinance.gov/data-research/consumer-credit-trends/student-loans/borrower-risk-profiles/)

5. Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. Journal of Banking & Finance, 34(11), 2767-2787. https://www.sciencedirect.com/science/article/abs/pii/S0378426610002372

6. Lenz, R. (2016). Peer-to-peer lending: Opportunities and risks. European Journal of Risk Regulation, 7(4), 688-700

7. myFICO. (n.d.). *What's in my FICO® Scores?* Retrieved November 20, 2024, from [https://www.myfico.com/credit-education/whats-in-your-credit-score](https://www.myfico.com/credit-education/whats-in-your-credit-score#:~:text=FICO%20Scores%20are%20calculated%20using,and%20credit%20mix%20(10%25))
