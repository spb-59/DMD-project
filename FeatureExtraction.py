
import os
import logging as lg
import numpy as np
from wfdb.io import rdrecord
from wfdb import Record
import pydmd as dmd
import multiprocessing as mp


path="processed1"


def featureExtract():

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

    num_workers = 4
    with mp.Pool(num_workers) as pool:
        pool.map(process_record, recordPath)

def process_record(record_path):
    try:
        recordSig = rdrecord(record_path)
        lg.info("Starting DMD for record: %s", record_path)
        
        features = extract(recordSig.p_signal)
        
        writeFile(features, recordSig.comments)
        
        lg.info("Finished extracting for record: %s", record_path)
    except Exception as e:
        lg.error("Error processing record %s: %s", record_path, e)

def extract(signal:np.ndarray):


    # Get the signals augumented
    signal=AugMat(signal.T,200)


    #fit the DMD model
    DMD=dmd.DMD()
    DMD.fit(signal)

    #get the eigenvalue and vectors
    eigs=DMD.eigs
    modes=DMD.modes

    #restack the modes to match the 12 leads
    restacked= modes.reshape(12, 200, -1).mean(axis=1)

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

    return f"{str(R_N)},{str(R_M)},{str(R_P)},{str(Lam_min)},{str(Lam_max)},{str(M_s)},{str(P_s)},{str(M_u)},{str(P_u)}\n"



def writeFile(features,comments):
    for comment in comments:
        file_path = os.path.join("features", f"{comment}.csv")
        
        # Print the comment for debugging
        print(comment)
        
        # Determine if the file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode
        with open(file_path, "a") as f:
            if not file_exists:
                # Write header if the file does not exist
                header = "R_N,R_M,R_P,Lam_min,Lam_max,M_s,P_s,M_u,P_u\n"
                f.write(header)
            
            # Write features
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








    





    



