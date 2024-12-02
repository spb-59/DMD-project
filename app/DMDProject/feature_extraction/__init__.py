import numpy as np
import pydmd as dmd


def extract(signal:np.ndarray):

    '''
    This function extracts the DMD features given  a signal
    '''
    # Get the signals augumented
    signal=aug_matrix(signal.T,200)


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

    #unstable lambda to all
    R_L=np.sum(np.abs(Lambda_u))/(np.sum(np.abs(Lambda_s))+np.sum(np.abs(Lambda_u)))

    R_M = np.sum(np.sum(np.abs(Pho_u), axis = 1))/(np.sum(np.sum(np.abs(Pho_u), axis = 1)) + np.sum(np.sum(np.abs(Pho_s), axis = 1)))
    R_P = np.sum(np.sum(np.angle(Pho_u), axis = 1))/(np.sum(np.sum(np.angle(Pho_u), axis = 1)) +  np.sum(np.sum(np.angle(Pho_s), axis = 1)))


    Lam_min = np.min(np.abs(Lambda_s),initial = float('inf') )
    Lam_max = np.max(np.abs(Lambda_u), initial =-float('inf') )

    #for unstble modes
    M_u = np.mean(np.abs(Pho_u), axis=1)
    P_u = np.mean(np.angle(Pho_u-np.angle(Pho_u[0])), axis=1)


    #for stable modes
    M_s = np.mean(np.abs(Pho_s), axis=1)
    P_s = np.mean(np.angle(Pho_s-np.angle(Pho_s[0])), axis=1)



    return {

    'R_N': np.nan_to_num(R_N, nan=0.0),
    'R_L': np.nan_to_num(R_L, nan=0.0),
    'R_M': np.nan_to_num(R_M, nan=0.0),
    'R_P': np.nan_to_num(R_P, nan=0.0),
    'Lam_min': np.nan_to_num(Lam_min, nan=0.0),
    'Lam_max': np.nan_to_num(Lam_max, nan=0.0),
    'M_u': np.nan_to_num(M_u, nan=0.0),
    'P_u': np.nan_to_num(P_u, nan=0.0),
    'M_s': np.nan_to_num(M_s, nan=0.0),
    'P_s': np.nan_to_num(P_s, nan=0.0)
}


def aug_matrix(signal: np.ndarray, h: int):
    '''
    This function augments the matrix for applying DMD by a augmentation factor of input h
    '''
    n, m = signal.shape
    aug = []
    for i in range(n):
        for x in range(h):
            row = signal[i][x:m-h+x]
            aug.append(row)
    return np.vstack(aug)

def make_list(results:dict):
    '''makes a dictionary of features into a list for use in model'''
    processed=[]
    for name,item in results.items():
        if name not in ['M_u','P_u','M_s','P_s']:
            processed.append(item)
        else:
            for i in item:
                processed.append(i)
    
    return processed


