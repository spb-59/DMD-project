
import os
import logging as lg
import numpy as np
from wfdb.io import rdrecord
from wfdb import Record
import pydmd as dmd


path="physionet-data/processed1"


def featureExtract():
    lg.info("File paths extracted")
    recordPath=[]
    names=[]

    lg.info('Starting File Name extraction')

    for root, dirs, files in os.walk(path):

        for file in files:
            if file.endswith(".dat"):

                record_name = os.path.splitext(file)[0]
                

                record_path = os.path.join(root, record_name)

                names.append(record_name)

                recordPath.append(record_path)

    for record in recordPath:
        recordSig:Record=rdrecord(record)
        features=extract(recordSig.p_signal)
        print(features)
        writeFile(features,recordSig.comments)



def extract(signal:np.ndarray):


    # Get the signals augumented
    signal=AugMat(signal.T,300)


    #fit the DMD model
    DMD=dmd.DMD()
    DMD.fit(signal)

    #get the eigenvalue and vectors
    eigs=DMD.eigs
    modes=DMD.modes

    #restack the modes to match the 12 leads
    restacked= modes.reshape(12, 300, -1).mean(axis=1)

    #get lambda U for unstable S for stable
    Lambda_ind_u = np.where(np.abs(eigs) > 1)
    Lambda_ind_s = np.where(np.abs(eigs) < 1)

    Lambda_u = eigs[Lambda_ind_u] #unstable eigen values
    Lambda_s = eigs[Lambda_ind_s] #stable eigen values

    # Get the eigenvectors off the eigenvalue indexes
    Pho_u = restacked[:,Lambda_ind_u].reshape((12,Lambda_u.shape[0])) #unstable modes
    Pho_s = restacked[:,Lambda_ind_s].reshape((12,Lambda_s.shape[0])) #stable modes

    #number of Stable and unstable modes
    numS=Lambda_s.shape[0]
    numU=Lambda_u.shape[0]

    # unstable to stable DM ratio
    R_N=(numU)/(numU+numS)

    R_M = np.sum(np.sum(np.abs(Pho_u), axis = 1))/(np.sum(np.sum(np.abs(Pho_u), axis = 1)) + np.sum(np.sum(np.abs(Pho_s), axis = 1)))
    R_P = np.sum(np.sum(np.angle(Pho_u), axis = 1))/(np.sum(np.sum(np.angle(Pho_u), axis = 1)) +  np.sum(np.sum(np.angle(Pho_s), axis = 1)))


    Lam_min = np.min(np.abs(Lambda_s),initial = -1 )
    Lam_max = np.max(np.abs(Lambda_u), initial = -1 )

    M_s = np.sum(np.abs(Pho_s), axis=1)/numS
    P_s = np.sum(np.angle(Pho_s), axis=1) / numS


    M_u = np.sum(np.abs(Pho_u), axis=1)/numU 
    P_u = np.sum(np.angle(Pho_u-np.angle(Pho_u[0])), axis=1)/numU

    return f"{R_N},{R_M},{R_P},{Lam_min},{Lam_max},{M_s},{P_s},{M_u},{P_u}\n"



def writeFile(features,comments):
    for comment in comments:
        print(comment)
        exist=True
        if not os.path.exists(comment+".csv"):
            exist=False


        with open(comment+".csv","a") as f:
            if exist:
                f.write(features)
            else:
                f.write("R_N,R_M,R_P,Lam_min,Lam_max,M_s,P_s,M_u,P_u\n")
                f.write(features)

def AugMat(signal: np.ndarray, h: int):
    n, m = signal.shape
    aug = []
    for i in range(n):
        for x in range(h):
            row = signal[i][x:m-h+x]
            aug.append(row)
    return np.vstack(aug)



       

if __name__=="__main__":
    featureExtract()








    





    



