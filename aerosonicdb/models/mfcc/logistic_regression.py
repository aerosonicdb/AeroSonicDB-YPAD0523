import os
import json
import pickle
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from aerosonicdb.utils import get_project_root
from aerosonicdb.utils import fetch_k_fold_cv_indicies
from aerosonicdb.utils import load_frame_train_data

root_path = get_project_root()
train_path = os.path.join(root_path, 'data/processed/13_mfcc_5_train.json')
output_path = os.path.join(root_path, 'models/mfcc_logistic_regression')

if not os.path.exists(output_path):
    os.makedirs(output_path)


def run_cv(train_path=train_path, k=10, rand_seed=0):
    X, y, g = load_frame_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)

    results = cross_validate(model, X, y, cv=fetch_k_fold_cv_indicies(X, y, g, k=k), n_jobs=-1)
    mean = results['test_score'].mean() * 100
    st_dev = results['test_score'].std() * 100

    print(f'Acc: %.2f%% (%.2f%%)' % (mean, st_dev))

    return mean, st_dev


def train_save_model(train_path=train_path, output_path=output_path, filename='mfcc_logistic_regression.sav', rand_seed=0):
    
    X, y, g = load_frame_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)
    
    # Fit the model on training set
    model.fit(X, y)
    
    # save the model
    filename = os.path.join(output_path, filename)
    pickle.dump(model, open(filename, 'wb'))
    
    print(f'Model saved to: {filename}')


