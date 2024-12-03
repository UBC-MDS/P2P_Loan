# Logistic Regression Tuning
log_reg_param_dist = {
    "LogReg__C": np.logspace(-5, 5)
}

log_reg_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('LogReg', LogisticRegression(random_state=123, max_iter=20000))
])

log_reg_search = GridSearchCV(
    log_reg_pipe,
    param_grid=log_reg_param_dist,
    cv=10,
    n_jobs=-1,
    return_train_score=True
)

log_reg_search.fit(X_train, y_train)
cv_results = pd.DataFrame(log_reg_search.cv_results_)[[
    "rank_test_score",
    "param_LogReg__C",
    "mean_test_score",
    "mean_train_score"
]]

cv_results.sort_values(by="rank_test_score").head(5)