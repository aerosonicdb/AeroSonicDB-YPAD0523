import json
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


def load_data(data_path, target_labels):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data[target_labels]).astype(int)
    g = np.array(data['fold_label'])

    print("Data successfully loaded!")

    return X, y, g


def train_val_split(X, y, g, val_size=0.2):
    gs = GroupShuffleSplit(n_splits=2, test_size=val_size, random_state=0)
    train_index, val_index = next(gs.split(X, y, groups=g))

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    print(f'Length of X train: {len(x_train)}')
    print(f'Length of y train: {len(y_train)}')
    print(f'Length of X test: {len(x_val)}')
    print(f'Length of y test: {len(y_val)}')

    return x_train, y_train, x_val, y_val


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["sparse_categorical_accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_sparse_categorical_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def build_model(X):
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
        keras.layers.BatchNormalization(),

        # 1st dense layer
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        # 2nd dense layer
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        # output layer
        keras.layers.Dense(5, activation='softmax')
    ])

    # compile model
    optimizer = keras.optimizers.Adadelta()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def train_model(plot_training=False, save=False):

    # load data
    mfcc_path = '../../data/processed/13_mfcc_5_train.json'
    X, y, g = load_data(data_path=mfcc_path, target_labels='subclass_label')

    print(f'Length of X: {len(X)}')
    print(f'Length of y: {len(y)}')

    # stratified split of the training data
    X_train, y_train, X_val, y_val = train_val_split(X, y, g, val_size=0.2)

    model = build_model(X)

    model.summary()

    weights = {0: 0.7, 1: 7, 2: 2, 3: 0.5, 4: 25}

    if plot_training:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=260, epochs=100, class_weight=weights)

        plot_history(history)

    if save:
        # train model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=260, epochs=100, class_weight=weights)

        # save the model
        model.save('../../models/baseline_multiclass_mlp_mfcc/my_model')


if __name__ == '__main__':
    keras.utils.set_random_seed(0)
    train_model(plot_training=False, save=True)
