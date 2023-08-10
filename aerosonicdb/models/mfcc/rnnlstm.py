import json
import os
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GroupShuffleSplit
from aerosonicdb.utils import get_project_root
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


root_path = get_project_root()
train_path = os.path.join(root_path, 'data/processed/13_mfcc_5_train.json')
output_path = os.path.join(root_path, 'models')


def load_data(data_path, target_labels):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data[target_labels]).astype(int)
    g = np.array(data['fold_label'])

    print("Data successfully loaded!")

    return X, y, g


# function to return the cv splits by index
def fetch_cv_indicies(x, y, g):
    sgkf = StratifiedGroupKFold(n_splits=10)
    sgkf.get_n_splits(x, y)
    cv_splits = sgkf.split(x, y, g)

    return cv_splits


def train_val_split(x, y, g, val_size=0.1, rand_seed=0):
    gs = GroupShuffleSplit(n_splits=2, test_size=val_size, random_state=rand_seed)
    train_index, val_index = next(gs.split(x, y, groups=g))

    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    print(f'Length of X train: {len(x_train)}')
    print(f'Length of y train: {len(y_train)}')
    print(f'Length of X val: {len(x_val)}')
    print(f'Length of y val: {len(y_val)}')

    return x_train, y_train, x_val, y_val


def build_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def init_model(x):
    # create network
    input_shape = (x.shape[1], x.shape[2])
    model = build_model(input_shape)
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='BinaryCrossentropy',
                  metrics=['binary_accuracy'])

    # model.summary()

    return model


def run_cv(rand_seed=0):
    keras.utils.set_random_seed(rand_seed)
    X, y, g = load_data(data_path=train_path, target_labels='class_label')
    build = init_model(X)
    model = KerasClassifier(model=build, epochs=10, batch_size=216, random_state=rand_seed, verbose=2)

    results = cross_validate(model, X, y, cv=fetch_cv_indicies(X, y, g))
    mean = results['test_score'].mean() * 100
    st_dev = results['test_score'].std() * 100

    print(f"Acc: %.2f%% (%.2f%%)" % (mean, st_dev))

    return mean, st_dev


def train_save_model(dir_path=output_path,
                     mfcc_path=train_path,
                     filename='mfcc_rnn_lstm'):
    keras.utils.set_random_seed(0)
    X, y, g = load_data(data_path=mfcc_path, target_labels='class_label')

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    model = init_model(X)
    model.summary()

    # train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=216, epochs=10)

    # save the model
    model_path = os.path.join(dir_path, filename, 'model')
    model.save(model_path)


if __name__ == "__main__":
    train_save_model()
