import os
import pandas as pd
import numpy as np
import scipy.stats as stats

import env

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import seaborn as sns

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
    X_train = train.drop(columns='churn').reset_index(drop=True)
    y_train = train[['churn']].reset_index(drop=True)

    X_validate = validate.drop(columns='churn').reset_index(drop=True)
    y_validate = validate[['churn']].reset_index(drop=True)

    X_test = test.drop(columns='churn').reset_index(drop=True)
    y_test = test[['churn']].reset_index(drop=True)
    
    #rename encoded columns
    
    train.rename(columns={'dependents_encoded': 'has_dependents'})
    validate.rename(columns={'dependents_encoded': 'has_dependents'})
    test.rename(columns={'dependents_encoded': 'has_dependents'})
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def get_tree(train_X, validate_X, train_y, validate_y):
    '''get decision tree accuracy on train and validate data'''

    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5, random_state=123)

    #fit model on training data
    clf = clf.fit(train_X, train_y)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf.score(train_X, train_y)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(validate_X, validate_y)}")
    
##---------------------------Visualization--------------------------##
import matplotlib.pyplot as plt

def get_pie_churn(train):
    '''get pie chart for percent of churn within telco'''

    # set values and labels for chart
    values = [len(train.churn[train.churn == True]), len(train.churn[train.churn == False])] 
    labels = ['Churn','Non-Churn'] 

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%', colors=['#c0ffee', '#ffc1cc'])
    plt.title('Customers Churning within the Train dataset')
    plt.show()
    
def get_bar_senior(train):
    '''Creates a bar chart comparing senior citizen vs non-senior citizen churn'''
    plt.title("Senior Citizens and churn rate, 1= Senior Citizen")
    sns.barplot(x="senior_citizen", y="churn", data=train, palette = 'Pastel1')
    churn_rate = train.churn.mean()
    plt.axhline(churn_rate, label="churn_rate")
    plt.legend()
    plt.show()
    
def get_bar_dependents(train):
    '''Creates a bar chart comparing churn rate for those with dependents and those without'''
    plt.title("Dependents and churn, 1= Has Dependents")
    sns.barplot(x="dependents_encoded", y="churn", data=train, palette = 'Pastel1')
    churn_rate = train.churn.mean()
    plt.axhline(churn_rate, label="churn_rate")
    plt.legend()
    plt.show()
    
def get_bar_contract(train):
    '''Creates a bar chart comparing contract type and churn rate'''
    plt.title("Contract Type and churn")
    sns.barplot(x="contract_type", y="churn", data=train, palette='Pastel1')
    churn_rate = train.churn.mean()
    plt.axhline(churn_rate, label="churn_rate")
    plt.legend()
    plt.show()

def get_bar_partner(train):
    '''This chart visualizes the relationship between partner status and churn'''
    plt.title("Partner Status and churn, 1 = Has Partner")
    sns.barplot(x="partner_encoded", y="churn", data=train, palette='Pastel1')
    churn_rate = train.churn.mean()
    plt.axhline(churn_rate, label="churn_rate")
    plt.legend()
    plt.show()
    
def monthly_charges_md(train):
    ''' This functions creates a bar chart comparing mean churn rate of monthly charges
    for customers who have churned vs those who have not.
    '''
    # Subset the data into churn and not-churned status
    not_churned = train[train.churn == 0]
    churned = train[train.churn == 1]
    #assign values and labels
    values = [not_churned.monthly_charges.mean(), churned.monthly_charges.mean()]
    labels = ['not_churned', 'churned']
    # generate and display chart
    plt.bar(height=values, x=labels, color=['#ffc3a0', '#c0d6e4'])
    plt.title('Customer monthly charge amount differences in churn vs non-churn')
    plt.tight_layout()
    plt.show()
    
##-----------------------------Testing-----------------------------------##

def get_chi_senior(train):
    '''gets results of chi-square test for senior citizen and churn'''
    observed = pd.crosstab(train.churn, train['senior_citizen'])

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
def get_chi_dependents(train):
    '''gets results of chi-square test for dependents and churn'''
    observed = pd.crosstab(train.churn, train['dependents'])

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_partner(train):
    '''gets results of chi-square test for partner status and churn'''
    observed = pd.crosstab(train.churn, train['partner_encoded'])

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_chi_contract(train):
    '''gets results of chi-square test for contract type and churn'''
    observed = pd.crosstab(train.churn, train['contract_type'])

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
def get_t_monthly(train):
    "get t-test for monthly charges and churn"
    #Seperate samples into churn and not churn
    not_churned = train[train.churn == 0]
    churned = train[train.churn == 1]
    #Run t-test on these groups, variances are not equal
    t, p = stats.ttest_ind(not_churned.monthly_charges, churned.monthly_charges, equal_var=False)

    print(f't = {t:.4f}')
    print(f'p = {p:.4f}') 
    
def get_tree_test(X_train,X_test,y_train,y_test):
    '''get decision tree accuracy on train and test data'''
    #Testing on Decision Tree (depth of 3):
    tree = DecisionTreeClassifier(max_depth=3, random_state=123)
    # fitting the model(X, y)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    print(f"Accuracy of Decision Tree on train data is {tree.score(X_train, y_train)}")
    print(f"Accuracy of Decision Tree on test data is {tree.score(X_test, y_test)}")