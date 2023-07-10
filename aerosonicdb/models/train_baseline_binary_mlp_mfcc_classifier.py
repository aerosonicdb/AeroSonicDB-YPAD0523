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
    axs[0].plot(history.history["binary_accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_binary_accuracy"], label="test accuracy")
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
        keras.layers.Dropout(0.4),

        # 2nd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='BinaryCrossentropy',
                  metrics=['binary_accuracy'])

    return model


def train_model(plot_training=False, save=False):

    # load data
    mfcc_path = '../../data/processed/13_mfcc_5_train.json'
    X, y, g = load_data(data_path=mfcc_path, target_labels='class_label')

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    model = build_model(X)
    model.summary()

    # setup early stopping
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    if save:

        # train model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=260, epochs=100)

        # save the model
        model.save('../../models/baseline_binary_mlp_mfcc/my_model')

    if plot_training:

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=260, epochs=100, verbose=2)

        plot_history(history)


if __name__ == '__main__':
    keras.utils.set_random_seed(0)
    train_model(plot_training=False, save=True)
