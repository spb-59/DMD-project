
from collections import Counter
import os
from matplotlib import pyplot as plt
from wfdb.io import rdrecord
from wfdb import Record
import wfdb.processing as wd
import pandas as pd
from wfdb.io import wrsamp
from denoising import denoise
import logging as lg
import time


# Set up the logger configuration
lg.basicConfig(
    filename='signal_processing.log',  # Log to a file
    filemode='a',                      # Append to the log file
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=lg.INFO                      # Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
)

console = lg.StreamHandler()
console.setLevel(lg.INFO)
formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
lg.getLogger().addHandler(console)




basePath="./physionet-data/"

SNOMED = pd.read_csv(basePath+'ConditionNames_SNOMED-CT.csv')


mapping = pd.Series(SNOMED['Acronym Name'].values, index=SNOMED['Snomed_CT'].astype(str)).to_dict()


def parseConditions(comments):


    for data in comments:
        if data.startswith("Dx:"):
            dx_codes = data.split(": ")[1].split(",")

            mapped = [mapping.get(dx, f"Unknown Dx: {dx}") for dx in dx_codes]
            return mapped




start=  time.perf_counter()
lg.info('Starting extraction')

lg.info("Opening Records File")
with open(basePath+'RECORDS', 'r') as file:
    file_paths = [line.strip() for line in file.readlines()]


lg.info("File paths extracted")
recordPath=[]
names=[]

lg.info('Starting File Name extraction')
for path in file_paths:
    for root, dirs, files in os.walk(basePath+path):

        for file in files:
            if file.endswith(".mat"):

                record_name = os.path.splitext(file)[0]
                

                record_path = os.path.join(root, record_name)

                names.append(record_name)

                recordPath.append(record_path)
        
lg.info(f'File name info extracted,{len(names)} records found ')

lg.info("Starting signal processing")

freq=[]
for i in range(50):
    lg.info(f"Starting processing for {names[i]}")

    record:Record=rdrecord(recordPath[i])
    lg.info("Record extracted")
    
    record.p_signal=wd.normalize_bound(record.p_signal)
    signal=record.to_dataframe()
    sf=record.fs

    lg.info("Denoising starting")
    signal:pd.DataFrame=denoise(signal,sf)
    comments=parseConditions(record.comments)

    signal=signal.to_numpy()

    lg.info("Denoising finished for signal, creating entry to directory")

    wrsamp(record_name=names[i],fs=sf,units=record.units,sig_name=record.sig_name,p_signal=signal,fmt=record.fmt,comments=comments,write_dir='./physionet-data/processed1')
    lg.info("Entry created, starting next record")


lg.info("Processing Complete")
end=  time.perf_counter()
lg.info(f'Signal processing complete for {len(names)} records in {end-start:.3f} time')
counter = Counter(freq)









