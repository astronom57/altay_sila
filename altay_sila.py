#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:15:41 2022

altay - sila

@author: lisakov
"""



#import sys

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
np.set_printoptions(precision=4)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine import transformation as vt

import lightgbm as lgb

#import optuna.integration.lightgbm as lgbo
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) 



def read_data(file):
    column_names = ['ID', 'code_group', 'year_enter', 'sex', 'reason', 'language', 'birthday', 
                    'school', 'school_where', 'school_when', 'posobie', 'country', 'region', 'city', 
                    'dormitory', 'mother', 'father', 'country_parents', 'adopted', 'countryside', 
                    'foreigner', 'faculty', 'av_score', 
                    'status']
    df = pd.read_csv(file, names=column_names, header=0)
    df.drop('ID', axis=1, inplace=True)
    
    df.year_enter = df.year_enter.astype(int)
    
    return df

def read_test_data(file, keep_names=False):
    
    if keep_names is True:
        df = pd.read_csv(file, header=0)
        df['Год_Поступления'] = df['Год_Поступления'].astype(int)
    else:
        column_names = ['ID', 'code_group', 'year_enter', 'sex', 'reason', 'language', 'birthday', 
                        'school', 'school_where', 'school_when', 'posobie', 'country', 'region', 'city', 
                        'dormitory', 'mother', 'father', 'country_parents', 'adopted', 'countryside', 
                        'foreigner', 'faculty', 'av_score']
        df = pd.read_csv(file, names=column_names, header=0)
        df.year_enter = df.year_enter.astype(int)
        df.drop('ID', axis=1, inplace=True)

    return df



def transform_sex(dataframe):
    """ impute nans (only 7 records), fix some labels, transform to 0 and 1
    """
    df = dataframe.copy()
    #df = df.loc[~df.sex.isna()]
#    df.loc[:, 'sex'] = df.loc[:, 'sex'].str.lower()
#    df.loc[df.sex=='муж', 'sex'] = 0
#    df.loc[df.sex=='жен', 'sex'] = 1
    
    df.loc[(~df.sex.isna()) & (df.sex.str.contains('муж', flags=re.IGNORECASE)), 'sex'] = 0
    df.loc[(~df.sex.isna()) & (df.sex.str.contains('жен', flags=re.IGNORECASE)), 'sex'] = 1
    df.loc[df.sex.isna(), 'sex'] = df.sex.value_counts().nlargest(1).index.values[0]
    
    
    df.sex = df.sex.astype(int)
    return df

def itransform_sex(dataframe):
    df = dataframe.copy()
    df.loc[df.sex==0, 'sex'] = 'муж'
    df.loc[df.sex==1, 'sex'] = 'жен'
    return df


def fix_language(dataframe):
    """Fix labels in the language column"""
    df = dataframe.copy()
    df.loc[df.language.isna(), 'language'] = 'OTHER'
    
    df.loc[df.language.str.contains('англи*йский', flags=re.IGNORECASE), 'language'] = 'ENG'
    df.loc[df.language.str.contains('немецкий', flags=re.IGNORECASE), 'language'] = 'GER'
    df.loc[df.language.str.contains('французский', flags=re.IGNORECASE), 'language'] = 'FRA'
    df.loc[df.language.str.contains('русский', flags=re.IGNORECASE), 'language'] = 'RUS'
    return df


def fix_country(dataframe):
    df = dataframe.copy()
    cols = ['country', 'country_parents']
    
    for c in cols:
        df.loc[df[c].isna(), c] = 'OTHER'
        #df.loc[df[c].isna(), c] = 'Rus' <-- test this one too. Like the most frequent
        
        
        df.loc[df[c].str.contains('росс', flags=re.IGNORECASE), c] = 'Rus'
        df.loc[df[c].str.contains('каза[кх]', flags=re.IGNORECASE), c] = 'Kaz'
        df.loc[df[c].str.contains('китай', flags=re.IGNORECASE), c] = 'Chi'
        df.loc[df[c].str.contains('тадж', flags=re.IGNORECASE), c] = 'Tad'
        df.loc[df[c].str.contains('к[иы]рг[иы]', flags=re.IGNORECASE), c] = 'Kyr'
        df.loc[df[c].str.contains('армен', flags=re.IGNORECASE), c] = 'Arm'
        df.loc[df[c].str.contains('туркме', flags=re.IGNORECASE), c] = 'Tur'
        df.loc[df[c].str.contains('нигери', flags=re.IGNORECASE), c] = 'Nig'
        df.loc[df[c].str.contains('узбе', flags=re.IGNORECASE), c] = 'Uzb'
        df.loc[df[c].str.contains('франц', flags=re.IGNORECASE), c] = 'Fra'
        df.loc[df[c].str.contains('монго', flags=re.IGNORECASE), c] = 'Mon'
        df.loc[df[c].str.contains('укра', flags=re.IGNORECASE), c] = 'Ukr'
        df.loc[df[c].str.contains('молдо', flags=re.IGNORECASE), c] = 'Mol'
        df.loc[df[c].str.contains('герма', flags=re.IGNORECASE), c] = 'Ger'
    #    df.loc[df[c].str.contains('', flags=re.IGNORECASE), c] = ''

    return df

def fix_region(dataframe):
    df = dataframe.copy()
    
    most_frequent_region = df.region.mode()[0]
    df.loc[df.region.isna(), 'region'] = most_frequent_region
    
    df.loc[df.region.str.contains('алтай респ', flags=re.IGNORECASE), 'region'] = 'Республика Алтай'
    df.loc[df.region.str.contains('алайский', flags=re.IGNORECASE), 'region'] = 'Алтайский край'
    df.loc[df.region.str.contains('Алтайский крайай', flags=re.IGNORECASE), 'region'] = 'Алтайский край'
    df.loc[df.region.str.contains('ВКО', flags=re.IGNORECASE), 'region'] = 'Восточно-Казахстанская область'
    df.loc[df.region.str.contains('В-Казахстанская область', flags=re.IGNORECASE), 'region'] = 'Восточно-Казахстанская область'
        

    regex_pat = re.compile(r'обл\.*$', flags=re.IGNORECASE)
    df.region = df.region.str.replace(regex_pat, 'область', regex=True)
    
    regex_pat = re.compile(r'кр\.*$', flags=re.IGNORECASE)
    df.region = df.region.str.replace(regex_pat, 'край', regex=True)
    
    regex_pat = re.compile(r'респ\.*$', flags=re.IGNORECASE)
    df.region = df.region.str.replace(regex_pat, 'республика', regex=True)
    
    
    return df


def fix_city(dataframe):
    df = dataframe.copy()
    
    most_frequent_city = df.city.mode()[0]
    df.loc[df.city.isna(), 'city'] = most_frequent_city
    df.loc[df.city == ' ', 'city'] = most_frequent_city
    
    regex_pat = re.compile(r'\b\s*г\.*\s*\b', flags=re.IGNORECASE)
    df.city = df.city.str.replace(regex_pat, '', regex=True)
    regex_pat = re.compile(r'\b\s*с\.*\s*\b', flags=re.IGNORECASE)
    df.city = df.city.str.replace(regex_pat, '', regex=True)
    regex_pat = re.compile(r'\b\s*п\.*\s*\b', flags=re.IGNORECASE)
    df.city = df.city.str.replace(regex_pat, '', regex=True)
    
    return df






def fix_avscore(dataframe):
    """Av_score contains a mixture of normal EGE scores say [25:100], old school scores [0:5],
    and weird large scores [3000:5000]  (my guess: it is just an old score times 1000). I assume, that regardless of the scoring system kids are equally smart =>
    distribution of their scores should be the same. 
    
    The idea is to map [0:5] -> EGE, and [3000:5000] -> EGE
    
    Scores in the ranges [0:2], [6:29], [110:3000], [5000:inf] are flagged
    """
    # check  countries separately
    # Kaz: ege + old + outlier
    # Rus: ege + old +  weird
    # Kyr: ege + old
    # Chi: ???[30:100].  If ege, then their scores are concentrated around 40, which is low. 
    # Tad: ege + old +  weird
    # Arm: ege }
    # Tur: ege }
    # Nig: ege } all these have 1-2 students
    # Uzb: ege }
    # Fra: ege }
    # Mon: ege }
    # Ukr: ege }
    
    
    df = dataframe.copy()
    
       
    # impute data instead of dropping
    df.loc[df.av_score < 3, 'av_score'] = np.nan
    df.loc[df.av_score > 4000, 'av_score'] = np.nan
    df.loc[(df.av_score > 5) & (df.av_score < 30), 'av_score'] = np.nan
    df.loc[(df.av_score >= 110) & (df.av_score < 3000), 'av_score'] = np.nan
    
    
    ege_score = df.loc[(df.av_score > 29) & (df.av_score<110), 'av_score']  # 
    score_max = ege_score.max()
    score_min = ege_score.min()
    df.loc[(df.av_score >= 3) & ( df.av_score <= 5), 'av_score'] = scale_minmax(df.loc[(df.av_score >=3) & ( df.av_score <= 5), 'av_score'], score_min, score_max)
    df.loc[(df.av_score >= 3000) & ( df.av_score <= 5000), 'av_score'] = scale_minmax(df.loc[(df.av_score >= 3000) & ( df.av_score <= 5000), 'av_score'], score_min, score_max)
    
    
    # add a 'missing score' column
    df.loc[:, 'missing_av_score'] = 1
    df.loc[:, 'missing_av_score'] = df.where(df.av_score.isna(), 0)
    df.loc[df.missing_av_score == 1, 'av_score'] = np.random.normal((score_max + score_min) / 2, (score_max - score_min) / 3)
    
    
    return df

def scale_minmax(x, xmin, xmax):
    """homemade organic min-max scaler
    """
    return xmin + (x - min(x)) * (xmax - xmin) / (max(x)  - min(x))

def fix_foreigner(dataframe):
    """Assume that country is correct to fillna in the foreigner column
    """
    df = dataframe.copy()
    df.loc[(df.country=='Rus') & (df.foreigner.isna()), 'foreigner'] = 0
    df.loc[(df.country!='Rus') & (df.foreigner.isna()), 'foreigner'] = 1
    df.foreigner = df.foreigner.astype(int)
    return df
    
def fix_school(dataframe):
    """Not complete yet.
    So far just fixes nans"""
    
    df = dataframe.copy()
    df.loc[df.school.isna(), 'school'] = 'OTHER'
    df.loc[df.school.isna(), 'school'] = 'OTHER'
#    df.loc[df.school == 'NaN', 'school'] = 'OTHER'
    
    
    # I see two options: 
    # 1. assume that лицей, гимназия , сунц - better. They get 1, other get 0 
    # 2. calculate average av-score for school and put it as a weight. 
    
    # 1.
    df.loc[df.school.str.contains('лицей', flags=re.IGNORECASE), 'school'] = '1'
    df.loc[df.school.str.contains('гимназия', flags=re.IGNORECASE), 'school'] = '1'
    df.loc[df.school.str.contains('сунц', flags=re.IGNORECASE), 'school'] = '1'
    df.loc[df.school != '1', 'school'] = '0'
    df.school = df.school.astype(int)
    
    # 2.
    
    return df

def fix_school_where(dataframe):
    """Not complete yet.
    So far just fixes nans"""
    df = dataframe.copy()
    df.loc[df.school_where.isna(), 'school_where'] = 'OTHER'  # maybe replace with most frequent
    
    
    df.loc[df.school_where.str.contains('Барнаул', flags=re.IGNORECASE), 'school_where'] = 'Барнаул' 
    df.loc[df.school_where.str.contains('алтайский', flags=re.IGNORECASE), 'school_where'] = 'Алтайский край' 
    df.loc[df.school_where.str.contains('алтайского', flags=re.IGNORECASE), 'school_where'] = 'Алтайский край' 
    df.loc[df.school_where.str.contains('горно-алтайск', flags=re.IGNORECASE), 'school_where'] = 'Горно-Алтайск' 
    df.loc[df.school_where.str.contains('казах', flags=re.IGNORECASE), 'school_where'] = 'Казахстан' 
    
    
    
#    df.school_where = df.school_where.str.replace('.','', regex=False)
    
    regex_pat = re.compile(r'\b\s*г\.*\s*\b', flags=re.IGNORECASE)
    df.school_where = df.school_where.str.replace(regex_pat, '', regex=True)

    regex_pat = re.compile(r'\b\s*Россия\s*,*\s*\b', flags=re.IGNORECASE)
    df.school_where = df.school_where.str.replace(regex_pat, '', regex=True)

    regex_pat = re.compile(r'\bобл\b', flags=re.IGNORECASE)
    df.school_where = df.school_where.str.replace(regex_pat, 'область', regex=True)

    
    
    return df


def fix_school_when(dataframe):
    """Not complete yet.
    So far just fixes nans.
    
    NaNs = most frequent value.  
    Should be applied on train and test data separately.
    """
    df = dataframe.copy()
    mean_school_when = int(df.loc[~df.school_when.isna(), 'school_when']. mean())
    df.loc[df.school_when.isna(), 'school_when'] = mean_school_when
    df.school_when = df.school_when.astype(int)
    return df



def add_school_uni_diff(dataframe):
    """Add column with difference between the year of entering uni and finishing school.
    """
    df = dataframe.copy()
    df.loc[:, 'school_uni_diff'] = df.year_enter - df.school_when
    return df
    
def fix_year_enter(dataframe):
    df = dataframe.copy()
    df.year_enter = df.year_enter.astype(int)
#    df.drop(df.loc[df.year_enter>2022].index, axis=0, inplace=True)
    df.loc[df.year_enter > 2022, 'year_enter'] = 2022
    return df

def fix_posobie(dataframe):
    df = dataframe.copy()
    df.drop('posobie', axis=1, inplace=True)
    return df

def fix_adopted(dataframe):
    df = dataframe.copy()
    df.drop('adopted', axis=1, inplace=True)
    return df


def fix_dormitory(dataframe):
    df = dataframe.copy()
    most_freq_dormitory = df.dormitory.mode()[0]
    df.loc[df.dormitory.isna(), 'dormitory'] = most_freq_dormitory
    df.dormitory = df.dormitory.astype(int)
    return df


def fix_birthday(dataframe):
    df = dataframe.copy()
    df.birthday = pd.to_datetime(df.birthday)
    return df

def add_age_enter(dataframe):
    """age on the 01 Sep of the year_enter
    """
    df = dataframe.copy()
    df.loc[:, 'date_enter'] = pd.to_datetime(df.year_enter.astype(str).str.cat(['_09_01']*df.year_enter.size), format='%Y_%m_%d')
    df.loc[:, 'age_enter'] = (df.date_enter - df.birthday) / np.timedelta64(1, 'Y')
    # maybe convert to int ? 
    
#    df.drop(df.loc[df.age_enter < 15].index, inplace=True)
    df.loc[df.age_enter < 15, 'age_enter' ] = 18
    
    df.drop('date_enter', axis=1, inplace=True)
    return df

def fix_mother(dataframe):
    df = dataframe.copy()
    df.mother = df.mother.astype(int)
    return df

def fix_father(dataframe):
    df = dataframe.copy()
    df.father = df.father.astype(int)
    return df    


def fix_countryside(dataframe):
    df = dataframe.copy()
    most_frequent_countryside = df.countryside.mode()[0]
    df.loc[df.countryside.isna(), 'countryside'] = most_frequent_countryside
    df.countryside = df.countryside.astype(int)
    return df


def fix_faculty(dataframe):
    """This feature is categorical, not numerical
    """
    df = dataframe.copy()
    df.faculty = df.faculty.astype(int).astype(str)
    return df

def transform_status(dataframe):
    df = dataframe.copy()
    df['status'].replace([4,3,-1, 5], [0,1,2,3], inplace=True)
    return df

def itransform_status(dataframe):
    df = dataframe.copy()
    df['status'].replace([0,1,2,3], [4,3,-1,5], inplace=True)
    return df

def fix(dataframe):
    df = dataframe.copy()
    
    if 'status' in df.columns:
        df = transform_status(df)
    
#    print(df.index.size)
    df = fix_posobie(df)
    df = fix_adopted(df)
#    print(df.index.size)
    df = transform_sex(df)
    df = fix_language(df)
#    print(df.index.size)
    df = fix_country(df)
    df = fix_region(df)
    df = fix_city(df)
    df = fix_foreigner(df)
#    print(df.index.size)
    df = fix_year_enter(df)
    df = fix_birthday(df)
    df = add_age_enter(df)
#    print(df.index.size)
    df = fix_mother(df)
    df = fix_father(df)
    df = fix_countryside(df)
#    print(df.index.size)
    df = fix_faculty(df)
    df.drop('birthday', axis=1, inplace=True)
    df.code_group = df.code_group.astype('O')
    
    
#    print('BEFORE FIXING AV_SCORE')
#    print(df.isna().sum())
    df = fix_avscore(df)
#    print('AFTER FIXING AV_SCORE')
#    print(df.isna().sum())
    return df

def fix2(dataframe):
    """These steps infer some parameters from the data (e.g. mean values). Therefore they should be done afterthe train-test split.
    """
    df = dataframe.copy()    
    df = fix_school(df)
    df = fix_school_when(df)
    df = fix_school_where(df)
    
    df = add_school_uni_diff(df)
    df = fix_dormitory(df)
    return df
    

def lgbm_f1_metric(y_hat, data):
    y_true = data.get_label()
    predictions = []
    for x in y_hat.reshape(3, -1).T:
        predictions.append(np.argmax(x))

    y_hat = np.array(predictions)
    return 'f1', f1_score(y_true, y_hat, average='macro'), True

def lgbm_acc_metric(y_hat, data):
    y_true = data.get_label()
    predictions = []
    for x in y_hat.reshape(3, -1).T:
        predictions.append(np.argmax(x))

    y_hat = np.array(predictions)
    return 'acc', accuracy_score(y_true, y_hat), True

def lgbm_auc_metric(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(3,-1)
    
#    print(y_hat.sum(axis=0))
    y_hat = y_hat / y_hat.sum(axis=0)
#    print(y_hat.sum(axis=0))

#    print(y_hat.T[0])
#    print(roc_auc_score(y_true, y_hat.reshape(3,-1), average='macro', multi_class='ovr'))
    return 'auc', roc_auc_score(y_true, y_hat.T, average='macro', multi_class='ovr'), True


def convert_multi_single(y_pred):
    predictions = []
    for x in y_pred:
        predictions.append(np.argmax(x))
    
    y_pred = np.array(predictions)
#    d = dict(zip([0,3,1,2], [4,5,3,-1]))
#    out = [d[i] for i in y_pred]
    return y_pred



#def get_f1(y_true, y_pred): #taken from old keras source code
#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#    precision = true_positives / (predicted_positives + K.epsilon())
#    recall = true_positives / (possible_positives + K.epsilon())
#    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
#    return f1_val

def custom_f1(y_true, y_pred):
    from tensorflow.keras import backend as K
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def equalize_classes(X_train, y_train, weights=None):
        # oversample to equalize the frequencies of classes
    # X_train and y_train have the same index. 
    # ?should it be done here or at earlier stages? 
    class_labels = [0,1,2]
    if weights is not None:
        if len(weights) != len(class_labels):
            raise ValueError
        # normalize weights so the lowest is 1
        weights = np.array(weights)
        weights = weights / weights.min()
    else:
        weights = np.ones(len(class_labels))

    
    target_number = y_train.value_counts().max()
    target_index = []
    for i,c in enumerate(class_labels):
        class_idx = np.array(y_train[y_train==c].index)
        class_number = class_idx.size
        new_class_idx = np.append(class_idx, np.random.choice(class_idx, target_number - class_number))  # only add randomly, not replace
        # add data according to weights
        if weights[i] > 1:
            new_class_idx = np.append(new_class_idx, np.random.choice(new_class_idx, int(new_class_idx.size*(weights[i] - 1))))
        
        
        target_index = np.append(target_index, new_class_idx)
    
    

    
    X_train = X_train.loc[target_index,:]
    y_train = y_train[target_index]

    return X_train, y_train



def transform_discretize(X_train, X_validate, X_test, col, bins, dtype=object):
    """ Take a numeric value 'col' and discretize it into bins. 
    Apply the same to X_train, X_validate, and X_test.
    
    Args:
        X_ (pd.dataFrame): X-data
        col (str): feature to transform
        bins ([float]): bins' boundaries. All data within [bins[i]: bins[i+1]] will be summed up
        
    Returns: X_train, X_validate, X_test
    """
    
    from feature_engine.discretisation import ArbitraryDiscretiser
    #bins = [14000, 16000, 18000, 19000, 20000]
    #cbins = np.digitize(X_train.code_group, bins)
    X_train.loc[:, col] = X_train.loc[:, col].astype(float)
    X_validate.loc[:, col] = X_validate.loc[:, col].astype(float)
    X_test.loc[:, col] = X_test.loc[:, col].astype(float)
    
#    user_dict = {'code_group': [0, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 40000]}
    user_dict = {col: bins}
    adis = ArbitraryDiscretiser(binning_dict=user_dict, return_object=False, return_boundaries=False)
    X_train = adis.fit_transform(X_train)
    X_validate = adis.transform(X_validate)
    X_test = adis.transform(X_test)
    
    X_train.loc[:, col] = X_train.loc[:, col].astype(dtype)
    X_validate.loc[:, col] = X_validate.loc[:, col].astype(dtype)
    X_test.loc[:, col] = X_test.loc[:, col].astype(dtype)
    
    return X_train, X_validate, X_test








##################### MAIN

# Transformation ideas
# + means that the data are fixed for further use (imputed, cleaned etc.)
#------------------------------
# + ID - drop
# + code_group - to bin 
# + year_enter
# + sex - [0,1]
# + reason - onehot
# + language  - [ENG, OTHER] ? Highly dominated by ENG
# + birthday - calculate age_enter
# school 
#    ГБОУ – государственное бюджетное образовательное учреждение (школа, лицей, гимназия, учебный комбинат, комплекс). Бывает так, что садик входит в школьный комплекс, тогда ГБДОУ – это подразделение ГБОУ.
#    МБОУ – муниципальное бюджетное образовательное учреждение (те же школы, лицеи и т. д., но с финансированием из бюджета города).
#    МКОУ – муниципальное казенное образовательное учреждение (опять те же самые школы всех видов с прямым финансированием из госбюджета, обычно городского, реже – регионального).
#    СОШ – средняя образовательная школа: обычная школа, без всяких затей. Как правило эта аббревиатура используется в сочетании с «МКОУ» – МКОУ СОШ.
#    МАОУ – муниципальное автономное образовательное учреждение. Ничего нового: это снова школы, гимназии, лицеи и так далее с государственным финансированием.
# not in dataset #    МКШ – малокомплектная школа. Расположены такие учебные заведения в сельской местности, в них учится немного учеников (несколько десятков), параллельных классов нет. Бывает и так, что какого-то класса вовсе нет, потому что в населенном пункте нет детей соответствующего возраста, которых надо учить. В совсем маленьких МКШ 8–10 учеников разного возраста могут сидеть в одном классе и заниматься вместе по всем предметам. Это «вымирающий вид» учебных заведений, за существование которого изо всех сил борются жители небольших сёл.
# not in dataset #   СУНЦ – специализированный учебный научный центр. Это школа-интернат для одаренных детей при высшем учебном заведении. Пример – знаменитый центр Колмогорова при МГУ.
#
# alternative categorizing (школа, лицей, гимназия, учебный комбинат, комплекс)
# https://schoolotzyv.ru/schools/9-russia/115-altajskij

# +- school_where - ?
# + school_when - before 2010, 2011, 2012 ... 
# + posobie - drop ? DROP
# + country - highly unbalanced towards Russia
# + region - makes sense only for Russia? Highly imbalanced towards Altai Krai
# + city - makes sense to Russia (major cities) ? or altaiskiy kray (barnaul or not )
# + dormitory - unbalanced. Nan -> most frequent
# + mother, father - suspiciously balanced. Maybe faked. Drop? 
# + country_parents - highly unbalanced towards Russia
# + adopted - highly unbalanced towards 0. Drop? Only 6 values of 1, all other are 0. DROP. 
# + countryside - [0,1], nan -> 1, unbalanced
# + foreigner - [0,1], nan -> 1, unbalanced
# + faculty - a bit unbalanced. 20 classes
# + av_score - how to convert grades to EGE score? outliers. 


# read train data, read test data
# fix obvious things in data. Impute etc.
# leave categorical variables for lightgbm to deal with
# cv and hyperparam tuning
# use the best model
# save predictions
TRAIN_FILE = '/home/lisakov/Programs/chempIIonat2022/altay2022/train_dataset_train.csv'
TEST_FILE = '/home/lisakov/Programs/chempIIonat2022/altay2022/test_dataset_test.csv'




df = read_data(TRAIN_FILE)
X_test = read_test_data(TEST_FILE)

df = fix(df)
X_test = fix(X_test)



X = df.drop('status', axis=1)
y = df['status']
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=57)
X = X_train.join(y_train)

X_train = fix2(X_train)
X_validate = fix2(X_validate)
X_test = fix2(X_test)



yj_transformer = vt.BoxCoxTransformer(variables = ['school_when'])
X_train = yj_transformer.fit_transform(X_train)
X_validate = yj_transformer.transform(X_validate)
X_test = yj_transformer.transform(X_test)



for f in ['av_score', 'school_when', 'age_enter', 'year_enter', 'school_uni_diff']:
    st_scaler = StandardScaler()
    X_train[f] = st_scaler.fit_transform(X_train[f].values.reshape(-1,1))
    X_validate[f] = st_scaler.transform(X_validate[f].values.reshape(-1,1))
    X_test[f] = st_scaler.transform(X_test[f].values.reshape(-1,1))

# bin some numeric or a-la numeric variables
X_train, X_validate, X_test = transform_discretize(X_train, X_validate, X_test, col='code_group', bins=[0, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 40000])
X_train, X_validate, X_test = transform_discretize(X_train, X_validate, X_test, col='school_uni_diff', bins=[-10] + list(np.arange(-1, 3, step=1) + 0.5) + [20], dtype=int)
X_train, X_validate, X_test = transform_discretize(X_train, X_validate, X_test, col='year_enter', bins=[-10] + list(np.arange(-3, 3, step=1) + 0.5) + [10], dtype=int)

rarelabel_encoder = RareLabelEncoder(tol=0.005, variables=['school_where'])
X_train = rarelabel_encoder.fit_transform(X_train)
X_validate = rarelabel_encoder.transform(X_validate)
X_test = rarelabel_encoder.transform(X_test)

rarelabel_encoder = RareLabelEncoder(tol=0.001, variables=['city', 'region', 'faculty'])
X_train = rarelabel_encoder.fit_transform(X_train)
X_validate = rarelabel_encoder.transform(X_validate)
X_test = rarelabel_encoder.transform(X_test)

rarelabel_encoder = RareLabelEncoder(tol=0.01, variables=[ 'country', 'country_parents', 'language'])
X_train = rarelabel_encoder.fit_transform(X_train)
X_validate = rarelabel_encoder.transform(X_validate)
X_test = rarelabel_encoder.transform(X_test)


cols2use = ['sex', 'year_enter', 'reason', 'language', 'code_group', 'av_score', 'faculty', 'age_enter', 'school_when', 'foreigner', 'school_uni_diff', 'school']
X_train = X_train.loc[:, cols2use]
X_validate = X_validate.loc[:, cols2use]
X_test = X_test.loc[:, cols2use]

onehot_encoder = OneHotEncoder(variables=['reason', 'language', 'faculty', 'code_group'])
X_train = onehot_encoder.fit_transform(X_train)
X_validate = onehot_encoder.transform(X_validate)
X_test = onehot_encoder.transform(X_test)
#



columns_cat = X_train.select_dtypes(include='int64').columns.values
for c in columns_cat:
    for XXX in [X_train, X_validate, X_test]:
#        XXX.loc[:, c] = XXX.loc[:, c].astype('category')
        pass
    
ids_of_categorical = [X_train.columns.get_loc(columns_cat[i]) for i in range(0,len(columns_cat))]
y_test_pred = None # to fill with predictions




deep = False # use DL or lightgbm

if deep == False:
    
    # oversample to equalize the frequencies of classes
    # X_train and y_train have the same index. 
    # ?should it be done here or at earlier stages? 
    do_equalize_classes = True
    if do_equalize_classes:
        X_train, y_train = equalize_classes(X_train, y_train, weights=[1,0.7,0.7])
    
    do_tune = False
  
    if do_tune == False:    
        naive = 2
        if naive == 2:
            # custom F1 -macro loss
            params = {
                "objective": "multiclassova",
                'metric': 'none',
                'num_boost_round':  700,
                'learning_rate': 0.02,
                'num_leaves': 160,
                'min_data_in_leaf': 10,
                'max_depth': 20,
                'max_bin': 400,
                'bagging_fraction': 0.75,
                'feature_fraction': 0.75,
                'bagging_freq': 5,
                "verbosity": 1,
                "boosting": "dart",                
                "seed": 57,
                'num_classes': 3
                }    

            evals_result = {}
            D_train = lgb.Dataset(X_train, label=y_train, categorical_feature=ids_of_categorical)
            D_validate = lgb.Dataset(X_validate, label=y_validate, categorical_feature=ids_of_categorical)
            
            clf = lgb.train(params, D_train, valid_sets=[D_validate], valid_names=['val'], feval=[lgbm_f1_metric, lgbm_auc_metric], evals_result=evals_result)
    
            lgb.plot_metric(evals_result, metric='f1')
            

            
            
        y_pred = clf.predict(X_validate)
        
        predictions = []
        for x in y_pred:
            predictions.append(np.argmax(x))
         
        y_pred = predictions
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_validate, y_pred, average='macro')))
    
        cm = confusion_matrix(y_validate, y_pred)
        fig, ax = plt.subplots(1,1)
        s = sns.heatmap(cm, fmt='d', annot=True)
        s.set_ylabel('True')
        s.set_xlabel('Predicted')


        lgb.plot_importance(clf)
        y_test_pred = clf.predict(X_test)
    
  
    if do_tune:
        print('DOING TUNING')
        # do the training
        # strategy: tune hyperparameters with optuna and use cross-validation
        
        D_train = lgb.Dataset(X_train, label=y_train)
        D_validate = lgb.Dataset(X_validate, label=y_validate)
        best_score = 0.5
    #    training_rounds = 10000
        
        def objective(trial):
            N_FOLDS = 5
            # Specify a search space using distributions across plausible values of hyperparameters.
            params = {
                "objective": "multiclassova",
                'metric': 'None',
                'num_boost_round': trial.suggest_int('num_boost_round', 200, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 30, 150, step=10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100, step=5),
                'max_depth': trial.suggest_int('max_depth', 15, 30),
#                'max_depth': 40,
#                'max_bin':  trial.suggest_int('max_bin', 10, 100, step=5),
                "max_bin": 10,
                'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.9, step=0.1),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 0.9, step=0.1),
    #            'bagging_fraction': 0.8, 
    #            'bagging_freq': trial.suggest_int('bagging_freq',1, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100, step=10),
                "verbosity": 1,
    #            "boosting": trial.suggest_categorical('boosting', ['dart', 'gbdt']),
                "boosting": 'dart',
                "seed": 57,
                'num_classes': 3,
                "feature_pre_filter": False,
                'categorical_feature': ids_of_categorical
            }
            
            # Run LightGBM for the hyperparameter values
            lgbcv = lgb.cv(params, D_train, feval=lgbm_f1_metric,
                           nfold=N_FOLDS, 
    #                       shuffle=True, stratified=True, 
    #                       verbose_eval=False,
    #                       return_cvbooster=True                       
                           )
            
            cv_score = lgbcv['f1-mean'][-1]
            print('F1 = {}'.format(cv_score))
            
            # Return metric of interest
            return cv_score
        
        
        
        study = optuna.create_study(direction='maximize')
    #    study.enqueue_trial(tmp_best_params)
        study.optimize(objective, timeout=3600*7)
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_slice(study)
        optuna.visualization.plot_param_importances(study)
        
        print('BEST parameters obtained within an OPTUNA tuning:\n')
        print(study.best_params)
        print(study.best_value)
        
        with open('/home/lisakov/Programs/chempIIonat2022/altay2022/optuna_lightgbm_params.txt','w') as outfile:
            print('OPTUNA optimized hyper parameters', file=outfile)
            print(study.best_params, file=outfile)
            print(study.best_value, file=outfile)
        
        
        
        # run the model with the best parameters
        
        params = study.best_params
        evals_result = {}
        D_train = lgb.Dataset(X_train, label=y_train)
        D_validate = lgb.Dataset(X_validate, label=y_validate)
        clf = lgb.train(params, D_train,  valid_sets=[D_validate], valid_names=['val'], feval=[lgbm_f1_metric, lgbm_acc_metric], evals_result=evals_result)
        lgb.plot_metric(evals_result, metric='f1')
                

predictions = []
for x in y_test_pred:
    predictions.append(np.argmax(x))

y_test_pred = np.array(predictions)
d = dict(zip([0,3,1,2], [4,5,3,-1]))
out = [d[i] for i in y_test_pred]


# write out the results
TEST = read_test_data(TEST_FILE, keep_names=True)
TEST.loc[:, 'Статус'] = out
TEST.loc[:, ['ID','Статус']].to_csv('/home/lisakov/Programs/chempIIonat2022/altay2022/prediction.csv', sep=',', index=False)








#
#
#
#
#        else:
#            N_FOLDS = 10
#            
#            params = {
#                "objective": "multiclassova",
#                'metric': 'None',
#                'num_boost_round':  1000,
#                'learning_rate': 0.1,
#                'num_leaves': 100,
#                'min_data_in_leaf': 15,
#                'max_depth': 50,
#                'max_bin': 10,
#                'bagging_fraction': 0.33,
#                'feature_fraction': 0.66,
#                'min_child_samples': 95,
#                'bagging_freq': 1,
#                "verbosity": 1,
#                "boosting": "dart",                
#                "seed": 57,
#                'num_classes': 3,
#                'categorical_feature': ids_of_categorical
#                }    
#       
#        
#            D_train = lgb.Dataset(X_train, label=y_train)
#            D_validate = lgb.Dataset(X_validate, label=y_validate)
#    #        clf = lgb.cv(params, D_train, num_boost_round=1000, nfold=N_FOLDS, shuffle=True, 
#    #                      stratified=True, verbose_eval=20, early_stopping_rounds=100, 
#    #                      return_cvbooster=True)
#            eval_results=[]
#            evals_result = {}
#
#            clf = lgb.cv(params, D_train, 
#                           feval=lgbm_f1_metric,
#                           nfold=N_FOLDS, shuffle=True, stratified=True, 
#                           verbose_eval=False, 
#                           return_cvbooster=True,
#                           )
#            
#            
#            
#            
#            eval_results.append(np.asarray(clf.evals_result_["valid_0"]["f1"])[:, np.newaxis])
#            
#            cv_results = np.hstack(eval_results)
#            print(cv_results)
#            
#            best_n_estimators = np.argmax(cv_results.mean(axis=1)) + 1
#            
#            model = lgb.LGBMClassifier(n_estimators=best_n_estimators)
#            model.fit(D_validate)
#            
#    #        print('{}-fold CV: AUC_mu mean values is {}'.format(N_FOLDS, np.mean(clf['auc_mu-mean'])))       
#        
#if deep:
#    # make a DL network
#    from tensorflow.keras.models import Sequential
#    from tensorflow.keras.layers import Dense, Dropout
#    from tensorflow.keras.regularizers import L2
#    import tensorflow as tf
#    import tensorflow_addons as tfa
#    from keras import initializers
#    
#    f1_metric = tfa.metrics.F1Score(num_classes=3, average='macro')
#    mcc_metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=3)
#
#    
#    
#    
#    def objective(trial):
#        first_dropout = trial.suggest_float('first_dropout', 0.2, 0.5, step=0.05)
#        second_dropout = trial.suggest_float('second_dropout', 0.2, 0.5, step=0.05)
#        
#        first_num_neurons = trial.suggest_int('first_num_neurons', 8, 64, step=4)
#        second_num_neurons = trial.suggest_int('second_num_neurons', 8, 64, step=4)
#
#        first_activation = trial.suggest_categorical('first_activation', ['relu', 'tanh'])
#        
#        first_regularization = trial.suggest_float('first_regularization', 1e-4, 1e-3)
#        
#        model = Sequential()
#        model.add(Dropout(first_dropout, input_shape=(X_train.shape[1],)))
#        model.add(Dense(first_num_neurons, activation=first_activation, kernel_regularizer=L2(first_regularization)))
#        model.add(Dropout(second_dropout, input_shape=(X_train.shape[1],)))
#        model.add(Dense(second_num_neurons, activation=first_activation, kernel_regularizer=L2(first_regularization)))
#        model.add(Dense(3, activation='softmax'))
#
#        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
#        
#        model.compile(loss='categorical_crossentropy',
#              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#              metrics=[f1_metric]
#              )
#        
#        number_epochs = trial.suggest_int('number_epochs', 50, 500, step=10)
#        batch_size = trial.suggest_int('batch_size', 1, 500, step=5)
#        history = model.fit(X_train, y_train_hot, validation_data=(X_validate, y_validate_hot), 
#                            class_weight=class_weight,
#                            epochs=number_epochs, batch_size=batch_size, verbose=2)
#
#        y_validate_predict = model.predict(X_validate)
#        f1_macro = f1_score(convert_multi_single(y_validate_predict), y_validate, average='macro')
#        
#        print('F1-macro = {:.3f}'.format(f1_macro))
#        
#        return f1_macro
#    
#    
#    # oversample to equalize the frequencies of classes
#    # X_train and y_train have the same index. 
#    # ?should it be done here or at earlier stages? 
#    do_equalize_classes = True
#    if do_equalize_classes:
#        X_train, y_train = equalize_classes(X_train, y_train)
#        
#        
#    BATCH_SIZE = 140
#    NEPOCHS = 200
##    BATCH_SIZE = 400
##    NEPOCHS = 1000
#    
#    # one-hot encode target
#    y_train_hot = pd.get_dummies(y_train)
#    y_validate_hot = pd.get_dummies(y_validate)
#    
#    # weights for classes
#    class_weight = {0:1, 1:0.7, 2:0.7}
##    class_weight = {0: 0.55, 1: 1.2, 2: 7.4}
#    for i in np.unique(y_train):
#        class_weight[i] = 1  / (y_train == i).sum() * y_train.index.size / len(class_weight)
#        
#    
#    do_tune = False
#    
#    if do_tune:
#        study = optuna.create_study(direction="maximize")
#        study.optimize(objective, timeout=3600*3)
#    
#        print("Number of finished trials: {}".format(len(study.trials)))
#    
#        print("Best trial:")
#        trial = study.best_trial
#    
#        print("  Value: {}".format(trial.value))
#    
#        print("  Params: ")
#        for key, value in trial.params.items():
#            print("    {}: {}".format(key, value))
#    
#    
#    else:
#        # my set of params:
##          Params: 
##            first_dropout: 0.5
##            first_num_neurons: 32
##            first_activation: relu
##            first_regularization: 0.001
##            learning_rate: 1e-4
##            number_epochs: 200
##            batch_size: 300
#        
##        Best trial:
##          Value: 0.7057270345416531
##          Params: 
##            first_dropout: 0.2
##            second_dropout: 0.25
##            first_num_neurons: 64
##            second_num_neurons: 36
##            first_activation: tanh
##            first_regularization: 0.00041577390175945425
##            learning_rate: 0.0013123151754725051
##            number_epochs: 420
##            batch_size: 141
#        
#        
#        model = Sequential()
#        model.add(Dropout(.2, input_shape=(X_train.shape[1],)))
#        model.add(Dense(64,  bias_initializer=initializers.HeNormal(), activation='tanh', kernel_regularizer=L2(4e-4)))
##        model.add(Dropout(.25))
##        model.add(Dense(8,  bias_initializer=initializers.Constant(0.24), activation='relu'))
#        model.add(Dropout(.25))
#        model.add(Dense(36,  bias_initializer=initializers.uniform(), activation='tanh'))
#        model.add(Dense(3, activation='softmax'))
#        
#        model.compile(loss='categorical_crossentropy',
#                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
##                      metrics=[ tf.keras.metrics.PrecisionAtRecall(0.8), tf.keras.metrics.RecallAtPrecision(0.8)]
#                      metrics=[f1_metric]
#                      )
#        
#        
#        history = model.fit(X_train, y_train_hot,
#                            validation_data=(X_validate, y_validate_hot), 
##                            validation_data=(X_train, y_train_hot), 
#                            class_weight=class_weight,
#                            epochs=NEPOCHS, batch_size=BATCH_SIZE, verbose=2)
#    
#        
#        model.summary()
#        fig, ax = plt.subplots(1,1, figsize=(15,10))
#        ax.set_ylim(0,1.5)
#        s1 = sns.lineplot(data=history.history['f1_score'], label='f1_score')
#        sns.lineplot(data=history.history['loss'], label='loss')
#        sns.lineplot(data=history.history['val_f1_score'], label='val_f1_score')
#        sns.lineplot(data=history.history['val_loss'], label='val_loss')
#        
#        y_validate_predict = model.predict(X_validate)
#        f1_macro = f1_score(convert_multi_single(y_validate_predict), y_validate, average='macro')
#        print('Final F1_macro score is {:.3f}'.format(f1_macro))    
#        print('MAX F1_macro validation score is {:.3f}'.format(np.max(history.history['val_f1_score'])))    
#    
#        cm = confusion_matrix(y_validate, convert_multi_single(y_validate_predict))
#        fig, ax = plt.subplots(1,1)
#        s = sns.heatmap(cm, fmt='d', annot=True)
#        s.set_ylabel('True')
#        s.set_xlabel('Predicted')
#        
#        
#    
#
#    y_test_pred = model.predict(X_test)
#    
#    
#    
#
#
#    
#
#
#
