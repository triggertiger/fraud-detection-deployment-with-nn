import pandas as pd
from sqlalchemy import create_engine, MetaData, update, Table, select
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv('.env')

# get data from sql database: 
class TrainDatesHandler:
    """
    starts and queries an SQLAlchemy engine.
    outputs data in different dataframes for the cases of - 
    initial training, for retraining on additional periods, and for predictions"""
    def __init__(self, date=None, username="user1", database=os.getenv('DATABASE')):
      
# data engine:
        self.database = database
        self.engine = create_engine(self.database, echo=False)
        self.dates_table = 'training_dates'
        self.users_table = 'users'
        self.transactions_table = 'transactions'
        self.username = username
        self.prediction_start_date = date
        

    @property 
    def dates_df(self):
        query = f'SELECT * FROM {self.dates_table}'
        return pd.read_sql(query, self.engine)
   
    @property 
    def date_for_new_training(self):
        """returns the newly updated date for retraining"""
        query = f"SELECT name, last_training_date FROM {self.users_table} WHERE name = '{self.username}';"
        users_df = pd.read_sql(query, self.engine)
        date = users_df['last_training_date'][0]
        return self.dates_df['train_date'][date]

    @property
    def training_date_index(self):
        """returns the index of the date at the user table for updating"""
        query = f'SELECT name, last_training_date FROM {self.users_table} WHERE name == "{self.username}";'
        users_df = pd.read_sql(query, self.engine)
        return users_df['last_training_date'][0]
        
    def update_db_last_train_date(self):
        """ updates the index of the training date in teh users
        table for the user. should be called before the retraining 
        function"""
        meta = MetaData()

        new_training_index = self.training_date_index + 1
        
        users = Table('users', meta, autoload_with=self.engine)
        update_stmt = (
            update(users).where(users.c.name == self.username).values(last_training_date=int(new_training_index) )
            )
        select_stmt = select(users).where(users.c.name == self.username)
                
        with self.engine.connect() as conn:
            conn.execute(update_stmt)
            res = conn.execute(select_stmt).all()
            conn.commit()
        return res
    
    def get_all_data(self):
        query = f'SELECT * FROM {self.transactions_table};'
        df = pd.read_sql(query, self.engine)
        df.drop(columns=['id', 'time_stamp', 'time_stamp_datetime'], inplace=True)
        return df

    def get_retraining_data(self):
        """gets the last trained data from the db through the class properties
        returns a df for training with data up to the following month"""
        last_data_date = pd.Timestamp(self.date_for_new_training)# + pd.DateOffset(months=1)
        datestring = last_data_date.strftime('%Y-%m-%d %H:%M:%S')
        
        query = f"SELECT * FROM {self.transactions_table} WHERE time_stamp_datetime < '{datestring}'::timestamp;"
        df = pd.read_sql(query, self.engine)
        df.drop(columns=['time_stamp_datetime'], inplace=True)

        return df

    def get_prediction_data(self, datestring='2019-01-01'):
        """gets date information from the app - only for prediction.
        returns dataframe for one month for prediction"""

        if self.prediction_start_date:
            if isinstance(self.prediction_start_date, datetime):
                date = self.prediction_start_date
            else: 
                date = datetime.strptime(self.prediction_start_date, '%Y-%m-%d') 
        else:
            date = datetime.strptime(datestring, '%Y-%m-%d') 

        year = date.year
        month = date.month 
        query = f'SELECT * FROM {self.transactions_table} WHERE year = {year} and month = {month};'        
        df = pd.read_sql(query, self.engine)
        df.drop(columns=['time_stamp_datetime'], inplace=True)

        return df
    

if __name__ == "__main__":
    user_data = TrainDatesHandler(date='2019-01-01')
    df = user_data.get_retraining_data()
    print(df.head())
    #user_data.update_db_last_train_date()

        


