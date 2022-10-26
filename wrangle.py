import os

import pandas as pd
import numpy as np

import env

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def get_db_url(db):
    '''
    Use your host, username, and password to codeup db to 
    acquire data from SQL
    '''
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return url

def get_telco_data():
    '''
    This function retrieves data from the codeup SQL db telco churn
    after joining the customers table with contract_type_id, internet_service_type_id,
    and payment_types table. If the file exists locally, it pulls data from the csv
    first. Finally, this functions returns a dataframe with all applicable data.
    '''
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
    
def prep_telco(df):
    '''
    This function takes in a dataframe (df) and returns a df suitable for exploration
    and modeling. It drops columns not needed for evaluation, drops null values, and converts
    total_charges to a float datatype. Additionally, it converts binary categorical values to
    numeric variables and creates dummy variables for non-binary categorical variables and 
    concatenates those series to the dataframe. Finally, this function returns a df.
    '''
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=False)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # Drop original churn variable (we encoded)
    df.drop(columns=['churn'], inplace=True)
    
    # Rename encoded churn
    df.rename(columns = {'churn_encoded':'churn'}, inplace=True)
    
    return df


def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 15% of the original dataset, validate is .1765*.85= 15% of the 
    original dataset, and train is 75% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.15, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.1765, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


def model_prep(train,validate,test):
    '''
    This function prepares train, validate, test for modeling by dropping columns not necessary
    or compatible with modeling algorithms.
    '''
    # drop unused columns 
    keep_cols = ['senior_citizen',
                 'dependents_encoded',
                 'monthly_charges',
                 'contract_type_Month-to-month',
                 'contract_type_One year',
                 'contract_type_Two year',
                 'churn']

    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]
    
    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    train_X = train.drop(columns='churn').reset_index(drop=True)
    train_y = train[['churn']].reset_index(drop=True)

    validate_X = validate.drop(columns='churn').reset_index(drop=True)
    validate_y = validate[['churn']].reset_index(drop=True)

    test_X = test.drop(columns='churn').reset_index(drop=True)
    test_y = test[['churn']].reset_index(drop=True)
    
    #rename encoded columns
    
    train.rename(columns={'dependents_encoded': 'has_dependents'})
    validate.rename(columns={'dependents_encoded': 'has_dependents'})
    test.rename(columns={'dependents_encoded': 'has_dependents'})
    
    return train_X, validate_X, test_X, train_y, validate_y, test_y