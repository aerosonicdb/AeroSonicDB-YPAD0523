import os
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_validate
from aerosonicdb.utils import get_project_root
from aerosonicdb.utils import fetch_k_fold_cv_indicies
from aerosonicdb.utils import load_train_data
from aerosonicdb.utils import train_val_split
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


root_path = get_project_root()
train_path = os.path.join(root_path, 'data/processed/13_mfcc_5_train.json')
output_path = os.path.join(root_path, 'models')


def build_model(x):
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),

        # 1st dense layer
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # 2nd dense layer
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='BinaryCrossentropy',
                  metrics=[tf.keras.metrics.AUC(curve='PR', name='PR-AUC')])

    return model


def run_cv(train_path=train_path, epochs=1, batch_size=216, rand_seed=0, verbose=0, k=10):
    keras.utils.set_random_seed(rand_seed)
    X, y, g = load_train_data(data_path=train_path, target_labels='class_label')
    build = build_model(X)
    model = KerasClassifier(model=build, epochs=epochs, batch_size=batch_size, random_state=rand_seed, verbose=verbose)

    results = cross_validate(model, X, y, cv=fetch_k_fold_cv_indicies(X, y, g, k=k), scoring=['average_precision',
                                                                                              'precision_macro',
                                                                                              'precision_micro'])

    mean = results['test_average_precision'].mean() * 100
    st_dev = results['test_average_precision'].std() * 100
    macro_mean = results['test_precision_macro'].mean() * 100
    macro_st_dev = results['test_precision_macro'].std() * 100
    micro_mean = results['test_precision_micro'].mean() * 100
    micro_st_dev = results['test_precision_micro'].std() * 100

    print(f'Average Precision Score: %.2f%% (%.2f%%)' % (mean, st_dev))
    print(f'Macro Precision Score: %.2f%% (%.2f%%)' % (macro_mean, macro_st_dev))
    print(f'Micro Precision Score: %.2f%% (%.2f%%)' % (micro_mean, micro_st_dev))

    return mean, st_dev


def train_save_model(output_path=output_path,
                     train_path=train_path,
                     filename='mfcc_mlp', 
                     epochs=1, 
                     batch_size=216, 
                     verbose=0,
                     rand_seed=0):
    
    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_labels='class_label')

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    model = build_model(X)
    model.summary()

    # train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=verbose)

    # save the model
    model_path = os.path.join(output_path, filename, 'model')
    model.save(model_path)
    
    print(f'Model saved to {model_path}.\n')


if __name__ == '__main__':
    run_cv()
