import argparse
import pickle
import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from process_code import process_two_directories

def train(files, plagiat1, plagiat2, pkl_to_save):

    train_df1 = process_two_directories(files, plagiat1, random_shuffle=True)
    train_df2 = process_two_directories(files, plagiat2, random_shuffle=True)
    train_df = pd.concat([train_df1, train_df2], ignore_index=True)
    
    X_train, X_val, y_train, y_val = train_test_split(train_df.drop(columns=['is_plagiat']), train_df['is_plagiat'])
    
    # Подбор параметров с помощью Optuna
    model = cb.CatBoostClassifier(
        objcetive='LogLoss',
        colsample_bylevel = 0.09881826824810507,
        depth=6,
        boosting_type='Ordered',
        bootstrap_type='Bernoulli',
        subsample=0.3266699776764633,
    )

    model.fit(X_train, y_train, plot=True, verbose=False, eval_set=(X_val, y_val), use_best_model=True)

    with open(pkl_to_save, 'wb') as f:
        pickle.dump(model, f)
    
    return 'Model has been saved'

parser = argparse.ArgumentParser(description='File to train model')
parser.add_argument('files', type=str, help='Directory for original files')
parser.add_argument('plagiat1', type=str, help='First directory for fake files')
parser.add_argument('plagiat2', type=str, help='Second directory for fake files')
parser.add_argument('--model', type=str, help='File to save model')
args = parser.parse_args()

files_dir = os.path(parser.files)
plagiat1_dir = os.path(parser.plagiat1)
plagiat2_dir = os.path(parser.plagiat2)

res = train(files_dir, plagiat1_dir, plagiat2_dir, args.model)
print(res)
