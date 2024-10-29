import numpy as np
import pandas as pd
    
def parse_german_data():
    """
    Parse the entire derman dataset
    """
    df = pd.read_csv("./data/german.csv", sep=";")
    df = df.dropna()
    df = df.replace({'?':np.nan}).dropna()
    
    numeric_df = df.select_dtypes(include=['int64']).copy()
    
    # Status of bank account
    status_mapping = {'no checking':0, '<0':1 , '0<=X<200':2, '>=200':3}
    numeric_df["Status of bank account"] = df["Status of bank account"].replace(status_mapping)

    # Savings
    saving_mapping = {"no known savings":0, "<100":1 , "100<=X<500":2, "500<=X<1000":3, ">=1000":4}
    numeric_df["Savings"] = df["Savings"].replace(saving_mapping)

    # Employment since
    employment_mapping = {"unemployed":0, "<1":1, "1<=X<4":2, "4<=X<7":3, ">=7":4}
    numeric_df["Employment since"] = df["Employment since"].replace(employment_mapping)

    # Job
    job_mapping = {"unemp/unskilled non res":0, "unskilled resident":1, "skilled":2, "high qualif/self emp/mgmt":3}
    numeric_df["Job"] = df["Job"].replace(job_mapping)
    
    # Binary variables

    tel_mapping = {'yes':1, 'none':0}
    numeric_df["Telephone"] = df["Telephone"].replace(tel_mapping)

    fw_mapping = {'yes':1, 'no':0}
    numeric_df["foreign worker"] = df["foreign worker"].replace(fw_mapping)

    type_mapping = {'good':1, 'bad':0}
    numeric_df["Type"] = df["Type"].replace(type_mapping)

    # Create gender variable
    numeric_df["Sex"] = df["Personal status and sex"].apply(lambda s: int(s.split()[0]=="male")) # 1 for male, 0 for female
    
    var_to_dummy = ["Credit History", "Purpose", 
                               "Other debtors / guarantors", "Property", 
                               "Other installment plans ", "Housing", "Job"]

    dummy_df = pd.get_dummies(df[var_to_dummy])
    
    final_df = pd.concat([numeric_df, dummy_df], axis=1, join="inner")
    
    final_df.to_csv("./data/german_parsed.csv", index=False)
    
    
    
def get_german_data(problem='reject_option', as_df=False):
    """
    problem: 'reject_option', 'alpha_risk', 'DP_aware', 'DP_unaware'
    """
    df = pd.read_csv("./data/german_parsed.csv")
    
    y = df['Type']
    df = df.drop('Type', axis=1)
    
    if (problem=='DP_unaware' or problem=='DP_aware'):
        
        S = df['Sex']
        df = df.drop('Sex', axis=1)
        
        if as_df:
            return df, S, y
        else:
            X = df.to_numpy() #features
            return X, S, y
    else:
        X = df.to_numpy()
        if as_df:
            return df, y
        else:
            X = df.to_numpy()
            return X, y
    
def parse_adult_data():
    """
    Parse the entire derman dataset
    """
    df = pd.read_csv("./data/adult.csv", )
    df = df.dropna()
    df = df.replace({'?':np.nan}).dropna()   
    
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    numeric_df = df.select_dtypes(include=['int64']).copy()
    
    workclass_mapping = {'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4,
                        'Local-gov':5, 'Without-pay':6, 'Never-worked':7}
    numeric_df['workclass'] = df['workclass'].replace(workclass_mapping)
    
    education_mapping = {'HS-grad':0, 'Some-college':1, '7th-8th':2, '10th':3, 'Doctorate':4,
                         'Prof-school':5, 'Bachelors':5, 'Masters':6, '11th':7, 'Assoc-acdm':8,
                         'Assoc-voc':9, '1st-4th':10, '5th-6th':11, '12th':12, '9th':13, 'Preschool':14}
    numeric_df['education'] = df['education'].replace(education_mapping)
    
    
    martial_status_mapping = {'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3,
                              'Married-civ-spouse':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6}
    numeric_df['marital.status'] = df['marital.status'].replace(martial_status_mapping)
    
    relationship_mapping = {'Not-in-family':0, 'Unmarried':1, 'Own-child':2, 'Other-relative':3,
                            'Husband':4, 'Wife':5}
    numeric_df['relationship'] = df['relationship'].replace(relationship_mapping)
    
    numeric_df.to_csv("./data/adult_parsed.csv", index=False)
    
    
def get_adult_data(problem='reject_option', as_df=False):
    """
    Parse the entire adult dataset
    problem: 'reject_option', 'alpha_risk', 'DP_aware', 'DP_unaware'
    """
    df = pd.read_csv("./data/adult_parsed.csv")
    
    y = df['income'] #target
    df = df.drop('income', axis=1)
    
    if (problem=='DP_unaware' or problem=='DP_aware'):
        
        S = df['sex']
        df = df.drop('sex', axis=1)
        
        if as_df:
            return df, S, y
        else:
            X = df.to_numpy() #features
            return X, S, y
    else:
        X = df.to_numpy()
        if as_df:
            return df, y
        else:
            X = df.to_numpy()
            return X, y

##############################################################

def get_frequencies(S):
    p = []
    for p_s in sorted(S.value_counts(1)):
        p.append(p_s)
    return p             
    
        
##############################################################

def drop_str(df):
    cols = df.columns
    for c in cols:
        if isinstance(df[c][1], str):
            column = df[c]
            df = df.drop(c, 1)
    return df

def log_numeric_features(df):
    cols = df.columns
    for c in cols:
        column =df[c]
        unique_values = list(set(column))
        n = len(unique_values)
        if n > 2:
            df[c] = np.log(1 + df[c])
            
###########################################################
def get_lawschool_data(as_df=False):
    
    df = pd.read_csv('./data/lawschool.csv')
    df = df.dropna()
    y = df['ugpa'] #target: gpa in [0,4]
    df = df.drop('ugpa', axis=1)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    df['race'] = [int(race == 7.0) for race in df['race']] #setting S=1 for white, S=0 for non-white
    S = df['race'] #sensitive attribute
    df = df.drop('race', axis=1)
    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y

def get_communities_data(as_df=False):
    
    df = pd.read_csv('./data/communities.csv')
    df = df.fillna(0)

    sens_attrs = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    df['race'] = df[sens_attrs].idxmax(axis=1) #creating a new column based on ethnicity
    df = df.drop(columns=sens_attrs)

    df = df.drop(df[df['ViolentCrimesPerPop']==0].index)
    y = df['ViolentCrimesPerPop'] #target
    df = df.drop('ViolentCrimesPerPop', axis=1)

    mapping = {'racePctWhite':1, 'racepctblack':0, 'racePctAsian':0, 'racePctHisp':0} 

    S = df['race'].map(mapping) #sensitive attribute: S=1 for white, S=0 for non-white
    df = df.drop('race', axis=1)

    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y
