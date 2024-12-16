
# P2P Online Lending Default Prediction- A Usecase on LendingClub Default Risk

#### Contributors: Mavis Wong, Yasmin Hassan, and Abeba Nigussie Turi

## About the Project
This work intends to leverage machine learning models to predict borrower behavior and, hence, the probability of default. More specifically, the work focuses on predicting loan defaults using historical data from the [Lending Club platform](https://github.com/matmcreative/Lending-Club-Loan-Analysis), specifically [this file](https://raw.githubusercontent.com/matmcreative/Lending-Club-Loan-Analysis/refs/heads/main/loan_data.csv). We uncover patterns and trends in borrower risk profiles by applying advanced preprocessing techniques, exploratory data analysis (EDA), and a Logistic Regression model. 

The final model demonstrated strong performance on unseen test data, achieving an accuracy of 84.0%. Out of 1,916 test cases, the model correctly predicted 1,608 cases, with 308 incorrect predictions. These errors included false positives (predicting a loan default when it didn't occur) and false negatives (failing to predict an actual default).

While false negatives pose a greater risk in financial decision-making, this model provides actionable insights to improve risk management and reduce potential financial losses for the platform. Despite its promising predictive capabilities, further research is needed to enhance the model's accuracy and better understand the characteristics of misclassified loans. Such improvements could be crucial in minimizing financial risks and maximizing the model's effectiveness in peer-to-peer lending platforms.

## Research Questions
1. How do borrower characteristics influence the probability of default in a P2P online lending settings?

2. What techniques can improve machine learning models for predicting defaults in the presence of class imbalance?

## Report
You can access the final report at [here](https://ubc-mds.github.io/P2P_Loan_Risk-Analysis/reports/p2p_lending_risk_analysis_report.html)


## Usage

### Set Up

1. Clone the Repository

 ```bash
   git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis
 ```

#### Option 1: Using environment.yml option

This is the recommended method to set up the environment that can allow you to run the file
  
1. Create the Conda environment:

 ```bash
    conda env create -f environment.yml
 ```

2. Activate the environment:

 ```bash
    conda activate loan_risk522
 ```

3. Verify the environment setup using this.

```bash
python -c "import pandas as pd; print('Environment set up successfully!')"
```

#### Option 2: Using the Docker Container Image

To use the containerized environment for this project, follow these steps if it is your first time.

1. Ensure you have Docker and Docker Compose installed.

2. Clone this repository and navigate to the root directory. 

3. Run in your terminal: 

```bash
   docker-compose up
```

4. Access the Jupyter Notebook interface by copying the URL that starts with `http://127.0.0.1:8888/lab?token=` in your terminal

5. In the terminal of JupyterLab, run the following code:

```bash
source ~/.bashrc
conda activate loan_risk522
```

<br>

### Running the Analysis

1. Open a terminal in Jupyter Lab or on your computer.
2. Navigate to the root directory of this project.
3. Run the following command to reset the project to a clean state:

```bash
make clean
```

4. Run the following commands in the terminal to recreate the entire analysis:

```bash
make all
```

<br>

### Clean Up - Docker

To shut down the container and clean up the resources, type Ctrl + C in the terminal where you launched the container, and then type:

```bash
   docker compose rm
```

## Developer notes
### Dependencies
- [Docker](https://www.docker.com)

- conda (version 23.9.0 or higher)

- conda-lock (version 2.5.7 or higher)

- mamba (version 1.5.8 or higher)

- nb_conda_kernels (version 2.3.1 or higher)

 - Python and packages listed in [here](https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/blob/main/environment.yml)


### Adding a new dependency and Updating Environment
1. Add the dependency to the environment.yml file on a new branch. If the package is pip installed, it should also be added to Dockerfile with the command:

`RUN pip install <package_name> = <version>`

2.  To update the `conda-linux-64.lock` file, run:

```bash
conda-lock -k explicit --file environment.yml -p linux-64
```

3. Re-run the scripts above using either option above.

4. If the environment.yml file is updated (e.g., new dependencies are added), you can update your existing environment with:

```bash
conda env update -f environment.yaml --prune
```

### Running the test suite
1. Run the analysis by following the instructions listed above. 
2. Open a terminal and navigate to the root directory of this project. 
3. Run the tests with the following command:

```bash
pytest
```

*More details about the test suite can be found in the [tests](https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/tree/main/tests) directory.*

## License
- **Code**:
If you are re-using/re-mixing, please provide attribution and link to this webpage. 
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
