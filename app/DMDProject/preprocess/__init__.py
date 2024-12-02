import pywt
from scipy import signal
import numpy as np
import pandas as pd
import logging

def removeBaselineWander(ecg_signal:pd.DataFrame,sf):
    
    sampling_rate = sf  
    cutoff_frequency = 0.8
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 

    b, a = signal.butter(1, cutoff_frequency / nyquist_rate, btype='highpass')
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal


def SWT(ecg_signal, wavelet,level):
   

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    for i in index:
        coeffs = pywt.swt(ecg_signal[i], wavelet, level=level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        uthresh = sigma * np.sqrt(2 * np.log(len(ecg_signal[i])))
        

        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        

        ecg = pywt.iswt(denoised_coeffs, wavelet)
        ecg_signal[i]=ecg[0].T
    
    return ecg_signal


def DWT(ecg_signal, wavelet,level):
   

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    for i in index:
        coeffs = pywt.wavedec(ecg_signal[i], wavelet, level=level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        uthresh = sigma * np.sqrt(2 * np.log(len(ecg_signal[i])))
        

        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        

        ecg_signal[i] = pywt.waverec(denoised_coeffs, wavelet)
    
    return ecg_signal

def notchFilter(ecg_signal, sf  ):
    notch_freq=50
    quality_factor=30
    sampling_rate =sf
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    b, a = signal.iirnotch(notch_freq / nyquist_rate, quality_factor)
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal

def removeHighFrequency(ecg_signal,sf):
    sampling_rate = sf  
    cutoff_frequency = 45
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 

    b, a = signal.butter(4, cutoff_frequency / nyquist_rate, btype='lowpass')
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal


def denoise(signal:pd.DataFrame,sf):
    '''
    This function combines the denoising functions and returns and signal with reduced noise.
    '''
    signal.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal.fillna(0,inplace=True)
    
    logging.info(msg="Starting denoising for current record")

    logging.info(msg="Baseline wander removal for current record")
    signal=removeBaselineWander(signal,sf)
    logging.info(msg="Baseline wander removal for current record complete")

    logging.info(msg="High Frequency removal for current record")
    signal=removeHighFrequency(signal,sf)
    logging.info(msg="High Frequency removal for current record complete")

    logging.info(msg="DWT removal for current record ")
    signal=DWT(signal,'sym6',6)
    logging.info(msg="DWT removal for current record complete")

    logging.info(msg="SWT removal for current record ")
    signal=SWT(signal,'bior4.4',3)
    logging.info(msg="SWT removal for current record complete ")

    logging.info('Denoising steps for current record complete')
    return signal
    
