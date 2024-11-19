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
    paths=paths
    
    MI = []
    HC = []
    DR=[]
    BBB=[]
    MH=[]
    VHD=[]
    MY=[]
    MSC=[]
    for path in paths:
        record = rdrecord(path)
        cond = parseConditions(record)

        if cond == 'HC':
            HC.append(path)
            lg.info(f'Classified {path} as Healthy Control')
        elif cond == 'MI':
            MI.append(path)
            lg.info(f'Classified {path} as Myocardial Infarction')
        elif cond == 'DR':
            DR.append(path)
            lg.info(f'Classified {path} as Dysrhythmia')
        elif cond == 'BBB':
            BBB.append(path)
            lg.info(f'Classified {path} as Bundle Branch Block')
        elif cond == 'MH':
            MH.append(path)
            lg.info(f'Classified {path} as Myocardial Hypertrophy')
        elif cond == 'VHD':
            VHD.append(path)
            lg.info(f'Classified {path} as Valvular Heart Disease')
        elif cond == 'MY':
            MY.append(path)
            lg.info(f'Classified {path} as Myocarditis')
        else:
            MSC.append(path)
            lg.info(f'Classified {path} as MSC')

    num_workers = mp.cpu_count()


    # lg.info(f'Starting processing with {num_workers} workers for HC...')
    # with mp.Pool(num_workers) as pool:
    #     pool.starmap(process_record, zip(HC[:50], ['HC'] * 50))

    # lg.info(f'Starting processing with {num_workers} workers for MI...')
    # with mp.Pool(num_workers) as pool:
    #     pool.starmap(process_record, zip(MI[:50], ['MI'] * 50))
    lg.info(len(MI)," THIS MUCH MI")
    conditions = [
        ('MI',MI),
        ('HC',HC),
        ('DR', DR),
        ('BBB', BBB),
        ('MH', MH),
        ('VHD', VHD),
        ('MY', MY),
        ('MSC', MSC)
    ]

    # Start processing each condition in parallel
    for cond_name, cond_list in conditions:
        lg.info(f'Starting processing with {num_workers} workers for {cond_name}...')
        with mp.Pool(num_workers) as pool:
            pool.starmap(process_record, zip(cond_list, [cond_name] * len(cond_list)))

        lg.info(f'Finished processing for {cond_name}.')

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
            write_dir='processedFullPaper'
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
        if 'Dysrhythmia' in c:
            return 'DR'
        if 'Bundle branch block' in c:
            return 'BBB'
        if 'Myocardial hypertrophy' in c:
            return 'MH'
        if 'Valvular heart disease' in c:
            return 'VHD'
        if 'Myocarditis' in c:
            return 'MY'
    return 'MSC'
        


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
