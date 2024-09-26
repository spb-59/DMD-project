import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import logging as lg
import time
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def cleanData(df:pd.DataFrame,i:int):
    df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df['Label']=i
    return df
    
def preprocess_data(X):
    # Convert all columns to numeric, forcing non-convertible values to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()  # Drop any rows with NaN values

    # Now handle object types
    for column in X.select_dtypes(include=['object']).columns:
        # Convert to strings if the column contains mixed types
        if not X[column].apply(lambda x: isinstance(x, str)).all():
            X[column] = X[column].astype(str)

        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    return X
def runModel():

    
    datasets=cleanData(pd.read_csv('features/SR.csv',low_memory=False),0).iloc[:5000]
    
    label=1
    for filename in os.listdir('features'):
    
        if filename.endswith('.csv') and filename != 'SR.csv' and not "Unknown" in filename and filename in ['AFIB.csv']:
        
            df = cleanData(pd.read_csv(os.path.join('features', filename),low_memory=False), 1)
            label+=1
            datasets=pd.concat([datasets,df])
    X = datasets.drop(["Label"], axis =1)
    
    Y = datasets["Label"]
    print(Y)



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=42)
  
    lg.info('Finished Splitting running model')
    start=time.perf_counter()
    XGBoostQ(X_tr=X_train, X_te=X_test, Y_tr=Y_train, Y_te=Y_test)
    end=time.perf_counter()

    lg.info('Ran for %d',end-start)

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
    ConfusionMatrixDisplay.from_estimator(estimator=rf_model, X=X_te, y=Y_te)

    # Classification Report
    print("Classification Report: Random Forest")
    print(classification_report(Y_te, rf_pred, digits=2))
    with open('results.txt', 'a') as f:
        f.write(classification_report(Y_te, rf_pred, digits=2))

# def XGBoost(X_tr, Y_tr, X_te, Y_te):
#     # Create the parameter grid
#     lg.info("Param tune")
#     param_grid = {
#         'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
#         'max_depth': range(1, 10),
#         'learning_rate': [0.01, 0.1, 0.2],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0],
#         'objective': ['multi:softmax']
#     }
#     lg.info("Param tune done")

#     lg.info("optimal param")
#     # GridSearchCV for hyperparameter tuning
#     optimal_params = GridSearchCV(
#         estimator=xgb.XGBClassifier(tree_method='gpu_hist', device='cuda', random_state=42),
#         param_grid=param_grid,
#         cv=10,
#         scoring='accuracy',
#         verbose=0,
#         n_jobs=1
#     )
#     lg.info("optimal param done")

#     # Fit the model
#     optimal_params.fit(X_tr, Y_tr)
#     print("Best parameters found: ", optimal_params.best_params_)

#     # Create the XGBoost model using the best parameters
#     best_params = optimal_params.best_params_
#     xgb_model = xgb.XGBClassifier(
#         tree_method='hist',
#         device='cuda',  # Ensure GPU utilization
#         random_state=42,
#         **best_params
     
#     )

#     # Fit the model
#     xgb_model.fit(X_tr, Y_tr)

#     # Predict the response
#     xgb_pred = xgb_model.predict(X_te)

#     # Plot confusion matrix
#     ConfusionMatrixDisplay.from_estimator(estimator=xgb_model, X=X_te, y=Y_te)

#     # Classification Report
#     print("Classification Report: XGBoost")
#     print(classification_report(Y_te, xgb_pred, digits=2))
#     with open('results.txt', 'a') as f:
#         f.write(classification_report(Y_te, xgb_pred, digits=2))



# if __name__=="__main__":
#     import xgboost as xgb
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split

#     # Load a simple dataset
#     data = load_iris()

#     X, y = data.data, data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create a simple XGBoost model with GPU support
#     xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', random_state=42)

#     # Fit the model and check for GPU utilization
#     xgb_model.fit(X_train, y_train)

def XGBoost(X_tr, Y_tr, X_te, Y_te):
    X_tr = preprocess_data(X_tr)
    X_te = preprocess_data(X_te)

    # Calculate class weight
    class_weight = compute_class_weight('balanced', classes=np.unique(Y_tr), y=Y_tr)
    lg.info("Starting hyperparameter optimization for XGBoost")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [class_weight[1]]
    }

    # Create the model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        device='cuda',  # Ensure GPU utilization
        random_state=42
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1',  # Use F1 score for evaluation
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    # Fit the model using GridSearchCV
    grid_search.fit(X_tr, Y_tr)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    lg.info(f"Best parameters found: {best_params}")
    lg.info(f"Best cross-validated F1 score: {best_score:.4f}")

    # Fit the model with the best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    # Predict the response
    xgb_pred = best_model.predict(X_te)

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    # Classification Report
    print("Classification Report: XGBoost")
    report = classification_report(Y_te, xgb_pred, digits=2)
    print(report)

def XGBoostQ(X_tr, Y_tr, X_te, Y_te):
    X_tr = preprocess_data(X_tr)
    X_te = preprocess_data(X_te)
    class_weight = compute_class_weight('balanced', classes=np.unique(Y_tr), y=Y_tr)
    lg.info("Fitting XGBoost model with predefined parameters")
    
    # Define the model with specific hyperparameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,        # Number of trees
        max_depth=5,             # Depth of each tree
        learning_rate=0.1,       # Step size shrinkage
        subsample=0.8,           # Fraction of samples to use for fitting
        colsample_bytree=0.8,    # Fraction of features to use for each tree
        objective='binary:logistic',  # Multi-class classification
        tree_method='hist',   # Use GPU
        device='cuda',            # Ensure GPU utilization
        random_state=42,
    
    )

    # Fit the model
    xgb_model.fit(X_tr, Y_tr)

    # Predict the response
    xgb_pred = xgb_model.predict(X_te)

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(estimator=xgb_model, X=X_te, y=Y_te)

    # Classification Report
    print("Classification Report: XGBoost")
    report = classification_report(Y_te, xgb_pred, digits=2)
    print(report)
