import os

import pandas as pd
import numpy as np

import env

def get_titanic_data():
    filename = "titanic.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env.get_db_url('titanic_db')
        df = pd.read_sql('SELECT * FROM passengers', url)
        df.to_csv("titanic.csv", index=False)
        return df

def get_iris_data():
    filename = "iris.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env.get_db_url('iris_db')
        df = pd.read_sql('SELECT * FROM measurements JOIN iris_db.species USING(species_id)', url)
        df.to_csv("iris.csv", index=False)
        return df


def get_telco_data():
    filename = "telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url =  env.get_db_url('telco_churn')
        df = pd.read_sql('SELECT * FROM customers\
        JOIN telco_churn.contract_types USING(contract_type_id)\
        JOIN telco_churn.internet_service_types USING(internet_service_type_id)\
        JOIN telco_churn.payment_types USING(payment_type_id)', url)
        df.to_csv("telco.csv", index=False)
        return df
    
