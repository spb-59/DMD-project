import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix



def runModel():
    Test = pd.read_csv('/home/senume/Project/MIS/mis-ECG_analysis_DMD/FEATURE_EXTRACTION/HODMD Paper/Features_Beat_Bundle branch block.csv')
    control = pd.read_csv('/home/senume/Project/MIS/mis-ECG_analysis_DMD/FEATURE_EXTRACTION/HODMD Paper/Features_Beat_Health Control.csv')

    Test = Test.dropna()
    control = control.dropna()

    Test_Label_Shape = Test.shape[0]
    Test_Label = np.ones((Test_Label_Shape)).tolist()
    Test.insert(len(Test.columns),"Label", Test_Label)

    control_Label_Shape = control.shape[0]
    control_Label = np.zeros((control_Label_Shape)).tolist()
    control.insert(len(control.columns),"Label", control_Label)

    DATASET = pd.concat([control, Test], ignore_index= True)
    X = DATASET.drop(["name", "Label"], axis =1)
    Y = DATASET["Label"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify= Y)



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
    with open('results.txt','a') as f:
        f.write(classification_report(Y_te, rf_pred, digits=2))
