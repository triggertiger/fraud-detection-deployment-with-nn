import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import logging
from utils.sql_data_queries import TrainDatesHandler
from utils.config import PARAMS, TRAIN_PARAMS, MLFLOW_URI, REGISTERED_MODEL_NAME, EXPERIMENT_NAME
import os
import numpy as np
import tempfile

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s"
)
# mlflow uri
mlflow.set_tracking_uri(uri=MLFLOW_URI)
logging.info(f'data_prep: tracking uri: {mlflow.get_tracking_uri()}')

class FraudDataProcessor:
    """
    Receives: dataframe with data from sql db. 
    SqlHandler provides one df for training from beginning (1/1/17) until the stated date, 
    and one prediction dataset for one month after the last training date (for example: 
    training < 1.1.19; prediction = JAN 2019), 
    and then handles the data pre-processing pipeline for training, retraining and predictions.
    Returns: Tensorflow datasets."""

    # pipeline instances:
    label_enc = LabelEncoder()
    # replace missing values with a constant text, then encode to numeric classes and scale
    state_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="online")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )
    # replace missing values with zero, then encode and scale
    zero_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ]
    )
    # implement number scaler on numerical features (no missing values on EDA)
    # implement text replacement to state and errors
    # implement zero replacement to zip, city and chip
    transformer = ColumnTransformer(
        transformers=[
            ("number_scaler", StandardScaler(), [0, 1, 2, 3, 4, 5, 7, 11, 13, 14]),
            ("NAN_replace_text", state_pipe, [9, 12]),
            ("NAN_replace_zero", zero_pipe, [6, 8, 10]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    def __init__ (self, date=None):
        # load data from sql db
        self.sql_handler = TrainDatesHandler(date=date)
        self.retrain_df = self.sql_handler.get_retraining_data()
        self.predict_df = self.sql_handler.get_prediction_data()
        self.ypred = self.predict_df['is_fraud']
        
        # save the date of last training, for splitting the data (last month is for prediction):
        #self.last_training_date = self.sql_handler.date_new_training
        #self.train_df = self.retrain_df.loc[self.retrain_df['time_stamp'] < self.last_training_date]#
        self.retrain_df.drop(columns=['id', 'time_stamp'], inplace=True)
        
        # save  a df of prediction dataset that will be presented to user
        self.present_df = self.predict_df.drop(columns=['time_stamp', 'is_fraud'])
        self.predict_df.drop(columns=['id', 'time_stamp', 'is_fraud'], inplace=True) 
                
    def x_y_generator(self):
        """ passes the training dataframe through pipeline to fit to the model:
        split to x, y, normalize data, label y and set bias."""

        # shuffle the data: 
        self.train_df = self.retrain_df.sample(frac = 1)

        # split to x, y
        xtrain = self.train_df.drop(columns=['is_fraud'])  
        ytrain = self.train_df[['is_fraud']]
                
        # apply transormer
        self.transformer.fit(xtrain)
        xtrain = self.transformer.transform(xtrain)
        ytrain = self.label_enc.fit_transform(ytrain)
        
        self.xpred = self.transformer.transform(self.predict_df)
        # for retraining evaluation purposes:
        ytest = self.label_enc.fit_transform(self.ypred)

        # set output bias
        neg, pos = np.bincount(self.label_enc.transform(self.retrain_df['is_fraud']))
        self.output_bias = np.log([pos / neg])

        #reshape labels tensor for tensorflow:
        logging.info(f'reshape ytrain to: {ytrain.shape}')
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)

        self.train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        self.test_ds = tf.data.Dataset.from_tensor_slices((self.xpred, ytest))
           
def update_params_output_bias(params, data: FraudDataProcessor):
    """update the output bias in the external params, 
    according to the data features for the purpose of training
    with the relevant output bias"""
    params['output_bias'] == data.output_bias    

def load_saved_model(version='latest'):
    start = time.time()
    
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"    
    model = mlflow.keras.load_model(model_uri=model_uri)
    logging.info(f'model loaded from: {model_uri}')

    logging.info(f'loading time: {time.time()-start}')
    return model

def load_model_weights(model, train_params):
    path = os.path.join(tempfile.mkdtemp(), 'initial_weights.weights.h5')
    train_params['initial_weights'] = path
    initial_weights = path
    logging.info(os.path.exists(path))
    
    if os.path.exists(path):
        logging.info(os.path.exists(path))
        logging.info("initial weights loaded")
        
    else: 
        model.save_weights(train_params['initial_weights'])        
        logging.info('weights saved')
        
    model.load_weights(initial_weights)
    return model

def model_re_trainer(model, data, params, train_params,exp_name=EXPERIMENT_NAME, output_bias_generator=True, callback=None):
    """ loads the model architecture for new training, with the new
    data for the relvant period"""
    mlflow.set_experiment(exp_name)
    with mlflow.start_run() as run:    
        if output_bias_generator:
            output_bias = tf.keras.initializers.Constant(params['output_bias']) 
        
        tags = {k: v for k, v in params.items()}
        mlflow.tensorflow.autolog(registered_model_name=REGISTERED_MODEL_NAME)
        mlflow.log_params(params)
        mlflow.log_params(train_params)
        mlflow.set_tags(tags)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc',
            verbose=0,
            patience=train_params['patience'],
            mode='max',
            restore_best_weights=True
        )
        callback_list = [early_stopping]

        # placeholder for Tensorboard if needed. no validation dataset in re-training
        if callback:
            callback_list.append(callback)
        
        if params['output_bias'] is None:
            model.layers[-1].bias.assign([0.0])
        
        train_ds = data.train_ds.batch(train_params['batch_size']).prefetch(2)
        test_ds = data.test_ds.batch(train_params['batch_size']).prefetch(1)

        model.fit(
            train_ds,
            batch_size=train_params['batch_size'],
            epochs=train_params['epochs'],
            callbacks=callback_list
        )
        print(model.summary())

        eval_results = model.evaluate(test_ds)
        for name, value in zip(model.metrics_names, eval_results):
            print(name, ': ', value)

        predictions = model.predict(data.xpred)
        
        def get_confusion_matrix(test_ds=test_ds):
            matplotlib.use('Agg')
            cm = ConfusionMatrixDisplay.from_predictions(
            np.concatenate([y for x, y in test_ds], axis=0), predictions > 0.2
        )
            mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')
            plt.close(cm.figure_)
        get_confusion_matrix()
    return eval_results 

def predict(model, data, threshold=0.5):
    """gets model predictions and returns in a df in a human readable format"""
    start = time.time()
    predictions = model.predict(data.xpred)
    labels = predictions >= threshold
    results_df = data.present_df
    results_df['is_fraud'] = labels
    logging.info(f'predicting time: {time.time() - start}')
    
    return results_df
  
def re_train_pipeline(date= '2019-01-01', model_version='latest'):
    
    data = FraudDataProcessor(date=date)
    data.x_y_generator()

    update_params_output_bias(PARAMS, data)
    model = load_saved_model(version=model_version)
    model = load_model_weights(model, PARAMS)
    eval_resutls = model_re_trainer(model, data, PARAMS, TRAIN_PARAMS)
    
    return eval_resutls

def predict_pipeline(date= '2019-01-01', model_version='latest'):
    logging.info(f'loading sql data for date {date}')
    start = time.time()
    data = FraudDataProcessor(date=date)
    data.x_y_generator()
    logging.info(f'sql loading & preprocessing time: {time.time()-start}')
    logging.info('loading model')
    model = load_saved_model(version=model_version)
    
    return predict(model, data)

    #data.x_y_generator()

if __name__ == "__main__":
# test predict pipeline
    # start = time.time()
    # results = predict_pipeline()
    # frauds = results[results['is_fraud']]
    # print(frauds.head(10))
    # print(frauds.shape)
    # logging.info(f'total elapsed time {time.time() - start}')
    # if frauds.shape[0] == 0:
    #     print('hurray! no fraud this month, you can go home')

# test retrain pipeline
    start = time.time()
    eval_resutls = re_train_pipeline()
    print(eval_resutls)
    logging.info(f'total elapsed time {time.time() - start}')

