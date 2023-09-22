import os
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score
from aerosonicdb.utils import get_project_root
from aerosonicdb.utils import fetch_k_fold_cv_indicies, train_val_split
from aerosonicdb.utils import load_train_data, load_test_data, load_env_test_data
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


ROOT_PATH = get_project_root()
FEAT_PATH = os.path.join(ROOT_PATH, 'data/processed')
TRAIN_PATH = os.path.join(FEAT_PATH, '13_mfcc_5_train.json')
TEST_PATH = os.path.join(FEAT_PATH, '13_mfcc_5_test.json')
ENV_FEAT_BASE = '_ENV_13_mfcc_5.json'
OUTPUT_PATH = os.path.join(ROOT_PATH, 'models')


def build_model(input_shape):

    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.Dropout(0.4))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.Dropout(0.4))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.Dropout(0.4))

    # flatten and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def init_model(x):
    # create network
    input_shape = (x.shape[1], x.shape[2], 1)
    model = build_model(input_shape)
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(curve='PR', name='PR-AUC')])

    # model.summary()

    return model


def run_cv(train_path=TRAIN_PATH,
           test_path=TEST_PATH,
           output_path=OUTPUT_PATH,
           epochs=1,
           batch_size=216,
           rand_seed=0,
           verbose=0,
           k=5,
           save_models=True):

    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_label='class_label')
    build = init_model(X)
    model = KerasClassifier(model=build, epochs=epochs, batch_size=batch_size,
                            random_state=rand_seed, verbose=verbose, class_weight='balanced')

    print(f'Running {k}-fold cross-validation...')

    results = cross_validate(model, X, y,
                             cv=fetch_k_fold_cv_indicies(X, y, g, k=k),
                             scoring='average_precision',
                             return_estimator=True)

    print('CV results:', results['test_score'], sep='\n')

    cv_mean = results['test_score'].mean() * 100
    cv_st_dev = results['test_score'].std() * 100

    print(f'Average Precision Score for {k}-fold CV: %.2f%% (%.2f%%)' % (cv_mean, cv_st_dev))

    cv_scores = (cv_mean, cv_st_dev, results['test_score'])

    cv_estimators = results['estimator']

    print(f'\nRunning {k}-model evaluation against Test set...')

    X_test, y_test = load_test_data(data_path=test_path, target_label='class_label')
    count = 1
    eval_results = []

    for est in cv_estimators:
        y_prob = est.predict_proba(X_test, batch_size=batch_size)[:, 1]
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        eval_results.append(ap_score)
        if save_models:

            # save the model
            model_path = os.path.join(output_path, f'cnn_{count}', 'model')

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            est.model_.save(model_path)
            count += 1

    print('Test evaluation results:', eval_results, sep='\n')

    test_mean = np.mean(eval_results) * 100
    test_st_dev = np.std(eval_results) * 100
    print(f'Average Precision Score against Test set: %.2f%% (%.2f%%)' % (test_mean, test_st_dev))

    test_scores = (test_mean, test_st_dev, eval_results)

    print(f'\nRunning {k}-model evaluation against the Environment set...')
    # evaluate against the environment set
    X_test, y_test = load_env_test_data(data_path=FEAT_PATH, json_base=ENV_FEAT_BASE,
                                        target_label='class_label')
    env_results = []

    for est in cv_estimators:
        y_prob = est.predict_proba(X_test, batch_size=batch_size)[:, 1]
        # print(y_test[:5])
        # print(y_prob.shape)
        # print(y_prob[:10])
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        env_results.append(ap_score)

    print('Environment evaluation results:', env_results, sep='\n')

    env_mean = np.mean(env_results) * 100
    env_st_dev = np.std(env_results) * 100
    print(f'Average Precision Score against Environment set: %.2f%% (%.2f%%)' % (env_mean, env_st_dev))
    env_scores = (env_mean, env_st_dev, env_results)

    return cv_scores, test_scores, env_scores


def train_save_model(output_path=OUTPUT_PATH,
                     train_path=TRAIN_PATH,
                     filename='mfcc_cnn',
                     epochs=1,
                     batch_size=216,
                     verbose=0,
                     rand_seed=0):
    
    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_label='class_label')

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    model = init_model(X)
    model.summary()

    # train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=verbose)

    # save the model
    model_path = os.path.join(output_path, filename, 'model')
    model.save(model_path)
    
    print(f'Model saved to {model_path}.\n')


if __name__ == '__main__':
    run_cv(epochs=50)
