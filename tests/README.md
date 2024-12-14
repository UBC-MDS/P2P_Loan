# Test suite developer notes
## Running the tests
1. Navigate to the root directory of the project.
2. Open a terminal and use the `pytest` command 
3. You should see an output similar to this if all test passed.

```bash
============================================== test session starts ==============================================
platform darwin -- Python 3.11.6, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/<user>/<path>/P2P_Loan_Risk-Analysis
plugins: typeguard-4.4.1
collected 65 items                                                                                              

tests/data_cleaning_test.py ....                                                                          [  6%]
tests/model_cv_test.py ....                                                                               [ 12%]
tests/validation_test.py ...................................................                              [ 90%]
tests/write_csv_test.py ......                                                                            [100%]

============================================== 65 passed in 2.55s ===============================================

```