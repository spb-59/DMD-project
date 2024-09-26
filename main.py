from Denoise import Denoise
from FeatureExtraction import featureExtract
import logging as lg
from Model import runModel


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


# Denoise()
featureExtract()
runModel()
