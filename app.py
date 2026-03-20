from flask import Flask, request, flash, jsonify, redirect, url_for, render_template, session
from flask_restful import Api 
import utils.data_prep_pipeline as data_prep_pipeline

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from utils.config import MLFLOW_URI, MLFLOW_REGISTERED_MODEL, DATABASE

from sqlalchemy import text, MetaData
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, length
from datetime import datetime
import os
import logging 
from utils.sql_data_queries import TrainDatesHandler
from dash import Dash, html, dash_table 
import mlflow
from mlflow import cli
import multiprocessing
from dotenv import load_dotenv
load_dotenv('.env')


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s"
)
# starting MLFlow server
#mlflow_client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)

mlflow.set_tracking_uri(MLFLOW_URI)
def start_mlflow_server():
    cli.server(["--host", "0.0.0.0", "--port", "5001"])  #"--host", "0.0.0.0", 
    
    
# starting Flask app and db connection, dash server
app = Flask(__name__)
app.secret_key = '123'

logging.info(os.getenv('DATABASE'))
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE') 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)
app.app_context().push()
bcrypt = Bcrypt(app)
api = Api(app)


# handling login
# 1. flask form
class LoginForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), length(min=2, max=20)],
        render_kw={"placeholder": "username"})
    password = PasswordField(
        validators=[InputRequired(), length(min=2, max=20)],
        render_kw={"placeholder": "Password"})
    submit = SubmitField('Log In')

# 2. Setting Loginmanager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    #return db.session.query(Users).get(int(user_id))
    return db.session.get(Users, int(user_id))

class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(64), nullable=False)
    last_training_date = db.Column(db.Integer)

# 3.2 setting auto-map for the rest of the db tables - 
metadata = MetaData()
metadata.reflect(bind=db.engine)

ReflectedBase = automap_base(metadata=metadata)
ReflectedBase.prepare(autoload_with=db.engine)

ReflectedBase.metadata.create_all(db.engine)
Dates = ReflectedBase.classes.training_dates
Transactions = ReflectedBase.classes.transactions


@app.route('/')
def home():
    title = "Fraud prediction portal, welcome"
    return render_template("index.html", title=title)

@app.route("/login", methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        
        user_logging_in = db.session.query(Users).filter_by(name = form.username.data).first()
        
        if user_logging_in:
            if bcrypt.check_password_hash(user_logging_in.password_hash, form.password.data):
                flash('Logged in successfully', 'info')
                login_user(user_logging_in)
                return redirect(url_for('userpage'))
            
    return render_template("login.html", form=form)
    
@app.route('/logout', methods=["POST", "GET"])
@login_required
def logout():
     logout_user()
     flash('You are logged out')
     return redirect(url_for("login"))

@app.route("/userpage", methods=["POST", "GET"])
@login_required
def userpage():

    # set name and date for headline
    username = current_user.name
    last_date_index = current_user.last_training_date + 1       
    last_train_date_row = db.session.query(Dates).filter(Dates.id==last_date_index).first()

    # set list of dates for dropdown:
    try:
        last_date_index = len(os.listdir(MLFLOW_REGISTERED_MODEL)) - 1      # this folder has all models + one yaml file
    
    except FileNotFoundError:
        return('under construction: model not yet set')

    datesrow = db.session.query(Dates.train_date).all()
    # unpack row tuples:
    alldates = [item[0] for item in datesrow]
    # show only available dates from the list:    
    dropdown_dates = alldates[:last_date_index]
    datestring = dropdown_dates[-1]
    
    # handle date selection: 
    selected_date = '2019-01-01'
    if request.method == 'POST':
        selected_date = request.form.get('dropdown')
        logging.info(f'selected_date is updated to: {selected_date}')
        
        # store in session
        dt = datetime.strptime(selected_date, '%Y-%m-%d')
        session['last_train_date'] = dt
        logging.info(f'updated date in sesison: {dt}')
        model_version_index = alldates.index(selected_date) + 1
        logging.info(f'model_version_index: {model_version_index}')
        # the index number will be also the serial number of the version to be used for prediction
        session['model_version'] = model_version_index
        return redirect(url_for('serve_dash_table'))

    return render_template('userpage.html', username=username, last_train_date=datestring, dates=dropdown_dates, selected_date=selected_date)

def predict_current_month(session=session):
    date = session['last_train_date']
    logging.info(f'date for prediction: {date}')
    session['predict_month'] = date.month
    session['predict_year'] = date.year
    model_version = session['model_version']
    logging.info(f'model version used: {model_version}')
    results = data_prep_pipeline.predict_pipeline(date=date, model_version=model_version)
    frauds = results[results['is_fraud']]
    
    return frauds

dash_app = Dash(__name__, server=app, url_base_pathname='/predict_current/')
dash_app.layout = html.Div([html.H2("Please log in to access predictions")])

@app.route("/predict", methods=["POST", "GET"])
@login_required
def serve_dash_table():
    headline_style={'textAlign': 'center'}
    frauds_df = predict_current_month()
    print('starting prediction')
    dash_app.layout = [
        html.Div([
            html.H1(
                children=f'Suspicious Transactions for the period: {session["predict_month"]}/{session["predict_year"]}',
                
                style=headline_style
            ),
            html.H3(f' there are {frauds_df.shape[0]} suspicious transactions for this month', style=headline_style),
            html.Br(),
        ]),
        html.Div([
            html.Div(children=[dash_table.DataTable(
                data = frauds_df.to_dict('records')
            )])
        ])
    ]
    return dash_app.index()

def update_last_training_date_in_db(session=session):
    
    sql_handler = TrainDatesHandler(username=current_user.name)
    new_date_row = sql_handler.update_db_last_train_date()
    new_date_index = new_date_row[0].last_training_date
    
    last_date_index = current_user.last_training_date + 1       
    last_train_date_row = db.session.query(Dates).filter(Dates.id==last_date_index).first()
    datestring = last_train_date_row.train_date
    
    dt = datetime.strptime(datestring, '%Y-%m-%d')
    session['last_train_date'] = dt
    session['predict_month'] = dt.month
    session['predict_year'] = dt.year
        
    return f'new date: {new_date_index}'

def train_new_data(session=session):
    date = session['last_train_date']
    eval_results = data_prep_pipeline.re_train_pipeline(date=date, model_version=session["model_version"])    
    
    # update session variable for next prediction: 
    session['model_version'] += 1
    return eval_results

@app.route('/retrain', methods=['POST', 'GET'])
def retrain():
    update_last_training_date_in_db()
    eval_results = train_new_data()
    return jsonify(eval_results)

if __name__ =="__main__":
        
    multiprocessing.set_start_method('fork')
    mlflow_process = multiprocessing.Process(target=start_mlflow_server)
    mlflow_process.start()
    app.run(debug=False, host="0.0.0.0", port=8080)
    mlflow_process.join()
    print('process terminated')
    