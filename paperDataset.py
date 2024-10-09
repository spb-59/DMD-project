import os
import logging as lg
from wfdb.io import rdrecord, Record
import multiprocessing as mp
import wfdb.processing as wd
from FeatureExtraction import featureExtract
from denoising import denoise
from wfdb.io import wrsamp
import uuid

recordPath = 'ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'

def preprocess():
    lg.info('Starting preprocessing...')
    file_paths = []
    with open(recordPath + "RECORDS", 'r') as file:
        file_paths = [line.strip() for line in file.readlines()]
   
    lg.info('Extracted file names from RECORDS')

    paths = [recordPath + path for path in file_paths]
    
    MI = []
    HC = []
    for path in paths:
        record = rdrecord(path)
        cond = parseConditions(record)
        if cond == 'HC':
            HC.append(path)
            lg.info(f'Classified {path} as Healthy Control')
        elif cond == 'MI':
            MI.append(path)
            lg.info(f'Classified {path} as Myocardial Infarction')

    num_workers = mp.cpu_count()
    lg.info(f'Starting processing with {num_workers} workers for MI...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(MI[:50], ['MI'] * 50))

    lg.info(f'Starting processing with {num_workers} workers for HC...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(HC[:50], ['HC'] * 50))

    lg.info('Preprocessing completed.')

def process_record(path, name):
    try:
        lg.info(f'Processing record {path}...')
        record = rdrecord(path)
        record.p_signal = wd.normalize_bound(record.p_signal)
        signal = record.to_dataframe()
        signal.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'vx', 'vy', 'vz']
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
            write_dir='processed2'
        )
        lg.info(f'Finished processing record {path}.')
    except Exception as e:
        lg.error(f'ERROR: {e}')  # Fixed logging error message

def parseConditions(record: Record):
    comment = record.comments
    for c in comment:
        if 'Myocardial infarction' in c:
            return 'MI'
        if 'Healthy control' in c:
            return 'HC'
    return 'NA'

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
