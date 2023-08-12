import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupShuffleSplit


def get_project_root() -> Path:
    return Path(__file__).parent.parent


# function to return the cv splits by index
def fetch_k_fold_cv_indicies(x, y, g, k=10):
    sgkf = StratifiedGroupKFold(n_splits=k)
    sgkf.get_n_splits(x, y)
    cv_splits = sgkf.split(x, y, g)

    return cv_splits


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


def load_frame_train_data(data_path, target_label):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data['mfcc'])
    y = np.array(data[target_label]).astype(int)
    g = np.array(data['fold_label'])

    print('Training data loaded.\n')

    X, y, g = explode_3_dims(x=X, y=y, g=g)

    return X, y, g


def load_train_data(data_path, target_labels):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data['mfcc'])
    y = np.array(data[target_labels]).astype(int)
    g = np.array(data['fold_label'])

    print('Training data loaded.\n')

    return X, y, g


def train_val_split(x, y, g, val_size=0.1, rand_seed=0):
    gs = GroupShuffleSplit(n_splits=2, test_size=val_size, random_state=rand_seed)
    train_index, val_index = next(gs.split(x, y, groups=g))

    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    print(f'Length of X train: {len(x_train)}')
    print(f'Length of y train: {len(y_train)}')
    print(f'Length of X val: {len(x_val)}')
    print(f'Length of y val: {len(y_val)}\n')

    return x_train, y_train, x_val, y_val