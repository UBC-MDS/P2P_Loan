# CHANGELOG- Feedback Checklist

## 1. README.md updated based on the peer review:
- Brocken hyperlink fixed
- General language check done with a 95% overall performance score using Grammarly business-expert-formal filters.
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/72db1742bcc27cd4480d08d2fbcdef5ccdfce618
- Research questions added 
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/93dfaab377d8f1597f16b1a131e296a49a8435ca
- Include link to raw data source in README
- Improve readability 
- Add more instructions for using environment.yml a user to compile the report locall
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/6a22f1d3b83a3107e419e75e7f6ec04c682b8088

## 2. Unit tests added
- https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/tree/main/tests
- Unit tests for data validation and model cross-evaluation
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/pull/56
- Unit tests for the write_csv.py 
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/3b11db8ac1750fb16905e510ea92968b7d1e694c

## 3. Final language check for the entire report done
- https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/74c3ba02e9259e37acdd86b51ee16841c2579ccb

## 4. Update model_tuning.py 
- Best C value in report output as a number instead of np.float64(0.000687)
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/204f466316415fce0f1d8d53fa5203ff543c9f9a

## 5. Modularize code into functions
- Function for data validation and model cross-evaluation
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/pull/56
- Function for the write_csv.py 
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/3b11db8ac1750fb16905e510ea92968b7d1e694c

## 6. Update environment.yml
- pip now has explicit version listed
- added pytest into environment.yml
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/efaeb4301d0fe3bd10cd3f390da141a02477d527

## 7. Update CONTRIBUTING.md 
- Clarify steps to contribute or report issues
- Added links to contributing guidelines and PEP8 stype guide
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/29f6365a653b9471c00659d85733e54c73035247

## 8. Update report.qmd
- Discuss the possiblity of using F1 score as classification metrics for further improvement of the model.
- Update Project structure for easy navigation
- Added cross-reference to Table 1: Feature Description
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/b623e4060706f6e94c2e5093d495b7f2f80482f0

## 9. Added README.md to tests/ folder
- Added details on running the tests.
    - https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/25a865977779f53ccc49860fa3fa3c2e53724f19

## 10. updated the docker image to our image hash pointing to the latest hash of the image
- https://github.com/UBC-MDS/P2P_Loan_Risk-Analysis/commit/1329e9fbeb0d5129f23b3a3750a0d2f630a6ce02