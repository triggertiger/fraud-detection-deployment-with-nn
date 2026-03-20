# Project Configurations
import pandas as pd
import os
from dotenv import load_dotenv
from tensorflow import keras

load_dotenv('.env')

# Paths
DATA_FOLDER = os.getenv('DATA_FOLDER')
ORIGINAL_CSV = "ibm_4y.csv"
DATA_FILE = "clean_cc_data.csv" 
DATABASE = "postgresql://inbalshalev@localhost:5432/fraud_transactions"

# MLFLOW
MLFLOW_URI = 'http://127.0.0.1:5001'
REGISTERED_MODEL_NAME = 'mlp_fraud_4'
MLFLOW_REGISTERED_MODEL = os.path.join('mlruns/models', REGISTERED_MODEL_NAME)
EXPERIMENT_NAME = 'mlp_fraud'
#DATABASE_FULL_PATH = f'sqlite:///{DATA_FOLDER}/fraud_transactions.db'

# model architecture paramseters:
PARAMS = {
    # logging: raise warning if the number of items in the list layer size 
    # is not smaller than the number of layers.
    'learning_rate': 1e-3,
    'output_bias': None,
    'dropout': 0.5,
    'train_feature_size': (15,),   #(xtrain.shape[-1],)),
    'layer_size': [32, 32, 32], #, 32],
    'activation1': 'relu',
    'nr_of_layers': 3
    }

# model metrics: 
MODEL_METRICS = {
    'binary_crossentropy': keras.metrics.BinaryCrossentropy(name='binary_crossentropy'), 
    'Brier_score': keras.metrics.MeanSquaredError(name='Brier_score'),
    'tp':keras.metrics.TruePositives(name='tp'),
    'fp':keras.metrics.FalsePositives(name='fp'),
    'tn':keras.metrics.TrueNegatives(name='tn'),
    'fn':keras.metrics.FalseNegatives(name='fn'), 
    'accuracy':keras.metrics.BinaryAccuracy(name='accuracy'),
    'precision':keras.metrics.Precision(name='precision'),
    'recall':keras.metrics.Recall(name='recall'),
    'auc':keras.metrics.AUC(name='auc'),
    'prc':keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve     
}

# training parameters
TRAIN_PARAMS = {
'patience': 10,
'epochs' : 50,
'batch_size': 2048,

}

TRAIN_DATES = {'dates': 
               pd.Series(pd.to_datetime(pd.date_range('2019-01-01','2020-02-01',freq='MS').strftime("%b-%y").tolist(), format='%b-%y'
))
}

USERS = [
        {'name': 'user1', 'password': os.getenv('USER1KEY'), 'training_date': 1},
         {'name': 'user2', 'password': os.getenv('USER2KEY'), 'training_date': 1}
         ]
