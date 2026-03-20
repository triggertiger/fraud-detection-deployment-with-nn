# Fraud Analysis Prediction and Automation
This is a show case project for developing and using the machine learning systems at the various stages of ML lifecycle.

- [description](#description)
- [instructions](#instructions)
- [structure and tools](#structure-and-tools)
- [Use Case, storyline and details](#use-case-storyline-and-details)

## Description
The core model is a neural network that predicts fraudulant credit card transactions, based on the [IBM Credit card Transactions Database](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions?resource=download&select=credit_card_transactions-ibm_v2.csv).
The model is automatically retrained on a monthly basis, based on new data, using a scheduled Github Actions workflow. Additionally, the project includes a UI that allows authorized users to log in and vie monthly predictions for potential fraud.  

## Structure and Tools
The project consists of the following steps, using different ML tools and frameworks: 
- **Exploratory Data Analysis (EDA)**: Primarily using Jupyter notebooks and Pandas
- **Data Cleaning and management**:Data is converted and stored as SQL, using Postgres DBMS
- **Data pre-processing**: Python functions and classes that manage data pre-processing pipeline
- **Modeling**: A binary-classifier neural network with Keras Sequential
- **Experiment Tracking and Model Registry** Model performance and registry is tracked using MlFlow
- **User interface (UI)**: Flask and Plotly-Dash-Table
- **Data storage** the data files are stored on Git LFS
- **Automation** using Github Actions workflow, triggered on a schedule. Currently set to daily

## Use Case, storyline and details: 
This project uses the IBM Credit Card Fraud Transactions Dataset, which contains 6 years of transactions with the labels Fraud or not_fraud. 4 years of data are used in the project, starting 1.1.2017, due to the size of the file.
The model is initially trained until 31.12.2018, and retrains on additional monthly data.

The UI includes viewing rights for two predefined users. The user credentials for the predictions-view are available only in the Project Presentation File that was submitted. 


## Usage Instructions:
1. Install Docker
2. Install git-lfs 
3. create a project folder in the desired location. for example: mkdir m2p_project
4. clone the repo: `git clone https://github.com/triggertiger/Model2Production.git' 
5. move into the project root directory
6. pull the data file `init_db.sql` file, using git-lfs: `git lfs pull`
6. Pull the images from Docker hub:
    b. `docker pull triggertiger/model_production:latest`
    a. `docker pull triggertiger/model_production:postgres` 
    (notice that `sudo` may be required in case that permission is denied)
7. Run: `docker compose -f docker-compose.yml up`
8. In the first run, building the database might take several seconds. In case it is returning an error, please give it a second run, by running:
`ctrl + c`, then `docker compose down` followed by `docker compose -f docker-compose.yml up`.
9. The app is running on localhost, on port 8080. Go to http://127.0.0.1:8080 in your browser, follow the screens and view the predictions. 
10. On finish: `ctrl + c`, then `docker compose down`.
11. For every subsequent run, both lines should be executed, to make sure you have the latest version:
    a. `docker pull triggertiger/model_production:latest`
    b. `docker pull triggertiger/model_production:postgres`
    c. `docker compose -f docker-compose.yml up`

## Reproduction: 

1. create a virtual environment and download the requirements
    `python3.9 -m venv model2production`
    `source model2production/bin/activate`
    `pip install -r requirements_mlflow.txt`

    **note:** make sure that tensorflow version is  <2.15, to avoid potential issues with MLFlow logging. 


2. Download the [dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv) from Kaggle and save it under `/data/6y_ibm.csv`
    > Alternatively, use the kaggle-cli tool for this with kaggle-cli tool (installed with the requirements), by running: 
    'kg dataset -u <username> -p <password> -o <owner> -d <dataset>`. 
3. Make sure you have Postgres running
4. Run: data/set_database.sh which will execute:
    - data/clean_csv.py
    - data/db_population.py
    - data/db_setup.py
5. In order to run experiments, make sure to update the environment variables in the .env file (see .env.example) and run: `python experiments_pipeline.py`
6. You're all set! 
    for experiments with new model parameters, run experiments_pipeline.py. 
    Continue as with the user instructions. 
