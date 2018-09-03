# Baseline


print "Linear Regression"
get_best_model_and_accuracy(lr, lr_params, X, y)
print "KNN"
get_best_model_and_accuracy(knn_pipe, knn_pipe_params, X, y)
print "Decision Tree"
get_best_model_and_accuracy(d_tree, tree_params, X, y)
print "Random Forest"
get_best_model_and_accuracy(forest, forest_params, X, y)
print "Gradient Boosting"
get_best_model_and_accuracy(gb, gb_params, X, y)
print "XGBoost"
get_best_model_and_accuracy(xg, xg_params, X, y)



LR : 
Best Accuracy: 0.86036036036
Best Parameters: {'penalty': 'l1', 'C': 1.0}

KNN
Best Accuracy: 0.846846846847
Best Parameters: {'classifier__n_neighbors': 7}

Decision Tree
Best Accuracy: 0.81981981982
Best Parameters: {'max_depth': 1}

Random Forest
Best Accuracy: 0.873873873874
Best Parameters: {'n_estimators': 1000, 'max_depth': 5}

Gradient Boosting 
Best Accuracy: 0.882882882883
Best Parameters: {'n_estimators': 100, 'max_depth': 1, 'min_samples_leaf': 2}

XGBoost
Best Accuracy: 0.882882882883
Best Parameters: {'colsample_bytree': 1, 'min_child_weight': 4, 'n_estimators': 50, 'subsample': 1, 'objective': 'binary:logistic', 'max_depth': 3}

# AFter Feature Selection based on Correlation

print "Linear Regression"
get_best_model_and_accuracy(lr, lr_params, X_subsetted, y)
print "KNN"
get_best_model_and_accuracy(knn_pipe, knn_pipe_params, X_subsetted, y)
print "Decision Tree"
get_best_model_and_accuracy(d_tree, tree_params, X_subsetted, y)
print "Random Forest"
get_best_model_and_accuracy(forest, forest_params, X_subsetted, y)
print "Gradient Boosting"
get_best_model_and_accuracy(gb, gb_params, X_subsetted, y)
print "XGBoost"
get_best_model_and_accuracy(xg, xg_params, X_subsetted, y)

Logistic Regression
Best Accuracy: 0.873873873874
Best Parameters: {'penalty': 'l2', 'C': 1.0}
KNN
Best Accuracy: 0.842342342342
Best Parameters: {'classifier__n_neighbors': 1}
Decision Tree
Best Accuracy: 0.842342342342
Best Parameters: {'max_depth': None}
Random Forest
Best Accuracy: 0.86036036036
Best Parameters: {'n_estimators': 500, 'max_depth': 5}
Gradient Boosting
Best Accuracy: 0.873873873874
Best Parameters: {'n_estimators': 100, 'max_depth': 1, 'min_samples_leaf': 1}
XGBoost
Best Accuracy: 0.887387387387
Best Parameters: {'colsample_bytree': 1, 'min_child_weight': 2, 'n_estimators': 1000, 'subsample': 1, 'objective': 'binary:logistic', 'max_depth': 1}

# After feautre selection with SelecktKB

Logistic Regression
Best Accuracy: 0.873873873874
Best Parameters: {'k_best__k': 11, 'classifier__C': 10.0, 'classifier__penalty': 'l1'}

KNN
Best Accuracy: 0.851351351351
Best Parameters: {'k_best__k': 18, 'classifier__n_neighbors': 7}

Decision Tree
Best Accuracy: 0.842342342342
Best Parameters: {'k_best__k': 12, 'classifier__max_depth': 5}

Random Forest
Best Accuracy: 0.869369369369
Best Parameters: {'k_best__k': 'all', 'classifier__n_estimators': 500, 'classifier__max_depth': 7}

Gradient Boosting
Best Accuracy: 0.887387387387
Best Parameters: {'classifier__min_samples_leaf': 2, 'k_best__k': 19, 'classifier__n_estimators': 500, 'classifier__max_depth': 1}

XGboost
Best Accuracy: 0.891891891892
Best Parameters: {'classifier__colsample_bytree': 1, 'classifier__subsample': 1, 'classifier__n_estimators': 50, 'classifier__min_child_weight': 4, 'k_best__k': 16, 'classifier__max_depth': 3, 'classifier__objective': 'reg:linear'}