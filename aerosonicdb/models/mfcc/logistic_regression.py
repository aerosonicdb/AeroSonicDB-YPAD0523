import os
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from aerosonicdb.utils import get_project_root

root_path = get_project_root()
train_path = os.path.join(root_path, 'data/processed/13_mfcc_5_train.json')
output_path = os.path.join(root_path, 'models/mfcc_logistic_regression')

if not os.path.exists(output_path):
    os.mkdir(output_path)


def explode_3_dims(x, y, g):
    x_expand = []
    y_expand = []
    g_expand = []

    for i in range(x.shape[0]):
        for n in range(x.shape[1]):
            x_expand.append(x[i][n])
            y_expand.append(y[i])
            g_expand.append(g[i])

    x_expand = np.array(x_expand)
    y_expand = np.array(y_expand)
    g_expand = np.array(g_expand)

    return x_expand, y_expand, g_expand


def load_train_data(data_path, target_label):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data[target_label]).astype(int)
    g = np.array(data['fold_label'])

    print("Train data successfully loaded!")

    X, y, g = explode_3_dims(x=X, y=y, g=g)

    return X, y, g


# function to return the cv splits by index
def fetch_cv_indicies(x, y, g):
    sgkf = StratifiedGroupKFold(n_splits=10)
    sgkf.get_n_splits(x, y)
    cv_splits = sgkf.split(x, y, g)

    return cv_splits


def run_cv(rand_seed=0):
    X, y, g = load_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)

    results = cross_validate(model, X, y, cv=fetch_cv_indicies(X, y, g), n_jobs=-1)
    mean = results['test_score'].mean() * 100
    st_dev = results['test_score'].std() * 100

    print(f"Acc: %.2f%% (%.2f%%)" % (mean, st_dev))

    return mean, st_dev


def save_model(filename='mfcc_logistic_regression.sav', rand_seed=0):
    X, y, g = load_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)
    # Fit the model on training set

    model.fit(X, y)
    # save the model to disk
    filename = os.path.join(output_path, filename)
    pickle.dump(model, open(filename, 'wb'))


