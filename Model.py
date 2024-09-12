import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix

def RF(X_tr, Y_tr, X_te, Y_te):
    # Create the param grid
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],  # Number of trees in the forest
        'max_depth': range(1, 10),  # Maximum number of levels in each tree
        'criterion': ['gini', 'entropy']  # Measure the quality of a split
    }

    # GridSearchCV to find the optimal parameters
    optimal_params = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=10,  # 10-fold cross-validation
        scoring='accuracy',
        verbose=0,
        n_jobs=-1
    )

    # Fit GridSearchCV
    optimal_params.fit(X_tr, Y_tr)
    print("Best parameters found: ", optimal_params.best_params_)

    # Extract the best parameters
    criterion = optimal_params.best_params_['criterion']
    max_depth = optimal_params.best_params_['max_depth']
    n_estimators = optimal_params.best_params_['n_estimators']

    # Create the Random Forest model with the optimal parameters
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    # Fit the model
    rf_model.fit(X_tr, Y_tr)

    # Predict the response
    rf_pred = rf_model.predict(X_te)

    # Plot confusion matrix
    plot_confusion_matrix(Y_te, rf_pred, display_labels=['Class 0', 'Class 1'])
    
    # Classification Report
    print("Classification Report: Random Forest")
    print(classification_report(Y_te, rf_pred, digits=2))
