
# P2P Online Lending Default Prediction- A Usecase on LendingClub Default Risk

#### Contributors: Mavis Wong, Yasmin Hassan and Abeba Nigussie Turi

## About the Project
This work intends to leaverage machine learning models to predict borrower behaviour and hence probability of default. More specifically, the work focuses aszxon predicting loan defaults using historical data from the Lending Club platform. By applying advanced preprocessing techniques, exploratory data analysis (EDA), and a Logistic Regression model, we uncover patterns and trends in borrower risk profiles. The final model demonstrated strong performance on unseen test data, achieving an accuracy of 84.0%. Out of 1,916 test cases, the model correctly predicted 1,608 cases, with 308 incorrect predictions. These errors included both false positives (predicting a loan default when it didn’t occur) and false negatives (failing to predict an actual default).
While false negatives pose a greater risk in financial decision-making, this model provides actionable insights to improve risk management and reduce potential financial losses for the platform. Despite its promising predictive capabilities, further research is needed to enhance the model's accuracy and better understand the characteristics of misclassified loans. Such improvements could play a crucial role in minimizing financial risks and maximizing the model’s effectiveness in peer-to-peer lending platforms.

Report
You can acces the final report at https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/blob/main/notebooks/p2p_lending_risk_analysis_report.ipynb


## How to Run the Analysis
1. **Clone the Repository**:
   ```bash
   git clone git@github.com:UBC-MDS/P2P_Loan_Risk-Analysis.git
   cd p2p-lending-risk-analysis

2. **Using environment.yml**

This is the recommended method to set up the environment that can allow you run the file
  Create the Conda environment:

    bash
    conda env create -f environment.yml
    
  Activate the environment:

    bash
    conda activate loan_risk522

## Using the Container Image
To use the containerized environment for this project follow this

1. Ensure you have Docker and Docker Compose installed.
2. Clone this repository and navigate to the root directory.
3. Run: docker-compose up
4. Access the Jupyter Notebook interface at http://localhost:8888.
5. Pull the Latest Image Pull the latest version of the container image from Docker Hub
(docker pull abeba/p2p_loan_risk_analysis:latest)

6. Run the Container Launch the container to start working on the project
(docker run -it --rm -p 8888:8888 abeba/p2p_loan_risk_analysis:latest)

6. Access your project files if you want to work with your local project files, mount the project directory


## Dependencies
  python=3.11.6
  -  numpy=1.24.4
  -  pandas=2.2.2
  -  scikit-learn=1.5.2
  -  altair=5.1.0
  -  matplotlib=3.9.2
  -  vegafusion-python-embed=1.6.9
  -  vegafusion=1.6.9
  -  vl-convert-python=1.6.0
  -  ipykernel=6.29.5
  -  pandera=0.20.4


## License
- **Code**: This project uses the MIT License. See [LICENSE.md]for details.

## Reference
1. Cai, S., Lin, X., Xu, D., & Fu, X. (2016). Judging online peer-to-peer lending behavior: A comparison of first-time and repeated borrowing requests. Information & Management, 53(7), 857-867.Consumer
2. Coşer, A., Maer-Matei, M. M., & Albu, C. (2019). PREDICTIVE MODELS FOR LOAN DEFAULT RISK ASSESSMENT. Economic Computation & Economic Cybernetics Studies & Research, 53(2).
3. Equifax. (n.d.). *Credit score ranges.* Retrieved November 20, 2024, from [https://www.equifax.com/personal/education/credit/score/articles/-/learn/credit-score-ranges/](https://www.equifax.com/personal/education/credit/score/articles/-/learn/credit-score-ranges/)
4. Financial Protection Bureau. (n.d.). *Borrower risk profiles: Student loans*. Retrieved November 20, 2024, from [https://www.consumerfinance.gov/data-research/consumer-credit-trends/student-loans/borrower-risk-profiles/](https://www.consumerfinance.gov/data-research/consumer-credit-trends/student-loans/borrower-risk-profiles/)
5. Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. Journal of Banking & Finance, 34(11), 2767-2787.
8. Lenz, R. (2016). Peer-to-peer lending: Opportunities and risks. European Journal of Risk Regulation, 7(4), 688-700
9. myFICO. (n.d.). *What's in my FICO® Scores?* Retrieved November 20, 2024, from [https://www.myfico.com/credit-education/whats-in-your-credit-score](https://www.myfico.com/credit-education/whats-in-your-credit-score#:~:text=FICO%20Scores%20are%20calculated%20using,and%20credit%20mix%20(10%25)
