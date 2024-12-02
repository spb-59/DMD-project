import pickle
from sklearn.discriminant_analysis import StandardScaler
import numpy as np


with open('./model/modelSVM.pkl', 'rb') as file:
        model = pickle.load(file)

def classify(features):
    X=np.array(features[:27])

    #TODO : get only significant features here
    X = X.reshape(1, 27)

    

    

    
    Y = model.predict(X) 

    return Y
    


    