import ast
import os
import logging as lg
import numpy as np
import pandas as pd
import wfdb
from wfdb.io import rdrecord, Record
import multiprocessing as mp
import wfdb.processing as wd
from FeatureExtraction import featureExtract
from denoising import denoise
from wfdb.io import wrsamp
import uuid

recordPath = 'ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'

def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

def preprocess():
    path="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    # Load and convert annotation data
    lg.info('Starting preprocessing...')
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    HC=[]
    AFIB=[]
    MI=[]
    for index, row in Y.iterrows():
        for key, val in row.scp_codes.items():  
            if key == 'NORM' and val == 100:
                HC.append(path+row.filename_hr)  
            elif key == 'AFIB' and val == 100:
                AFIB.append(path+row.filename_hr)
            elif key in['IMI','ASMI','ILMI',"AMI",'ALMI','LMI','IPLMI','IPMI','PMI'] and val == 100:
                MI.append(path+row.filename_hr)
    num_workers = mp.cpu_count()
    np.random.shuffle(MI)

    lg.info(f'Starting processing with {num_workers} workers for MI...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(MI[:50], ['MI'] * 50))



    lg.info('Preprocessing completed.')
        







def process_record(path, name):
    try:
        lg.info(f'Processing record {path}...')
        record = rdrecord(path)
        record.p_signal = wd.normalize_bound(record.p_signal)
        signal = record.to_dataframe()
        signal.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sf = record.fs
        signal.fillna(0, inplace=True)
        signal = denoise(signal, sf)
        
        output_name = str(uuid.uuid4())
        lg.info(f'Writing processed signal to file {output_name}...')
        
        wrsamp(
            record_name=output_name,
            fs=sf,
            units=record.units,
            sig_name=record.sig_name,
            p_signal=signal.to_numpy(),  # Convert DataFrame to values
            fmt=record.fmt,
            comments=[name],
            write_dir='processed5'
        )
        lg.info(f'Finished processing record {path}.')
    except Exception as e:
        lg.error(f'ERROR: {e}')  # Fixed logging error message


if __name__ == '__main__':
    # Configure logging to both file and console
    lg.basicConfig(
        level=lg.DEBUG,  # Set to DEBUG to capture all log levels
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    )

    # Create a file handler
    file_handler = lg.FileHandler('signal_processing.log')
    file_handler.setLevel(lg.DEBUG)  # Ensure this is set to DEBUG
    file_handler.setFormatter(lg.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a console handler
    console_handler = lg.StreamHandler()
    console_handler.setLevel(lg.DEBUG)  # Ensure this is set to DEBUG
    console_handler.setFormatter(lg.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Get the root logger and add handlers
    logger = lg.getLogger()
    logger.setLevel(lg.DEBUG)  # Set the logger level to DEBUG
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    preprocess()
    featureExtract()
