import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
from aerosonicdb.utils import get_project_root, fetch_k_fold_cv_indicies
from aerosonicdb.utils import load_flatten_train_data, load_flatten_test_data, load_flatten_env_test_data




ROOT_PATH = get_project_root()
FEAT_PATH = os.path.join(ROOT_PATH, 'data/processed')
TRAIN_PATH = os.path.join(FEAT_PATH, '13_mfcc_5_train.json')
TEST_PATH = os.path.join(FEAT_PATH, '13_mfcc_5_test.json')
ENV_FEAT_BASE = '_ENV_13_mfcc_5.json'
OUTPUT_PATH = os.path.join(ROOT_PATH, 'models')

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def run_cv(train_path=TRAIN_PATH,
           output_path=OUTPUT_PATH,
           k=5,
           rand_seed=0,
           test_path=TEST_PATH,
           save_models=True):

    X, y, g = load_flatten_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, solver='liblinear')
    print(f'Running {k}-fold cross-validation...')
    results = cross_validate(model, X, y,
                             cv=fetch_k_fold_cv_indicies(X, y, g, k=k),
                             scoring='average_precision',
                             n_jobs=-1,
                             return_estimator=True)

    print('CV results:', results['test_score'], sep='\n')

    cv_mean = results['test_score'].mean() * 100
    cv_st_dev = results['test_score'].std() * 100

    print(f'Average Precision Score for {k}-fold CV: %.2f%% (%.2f%%)' % (cv_mean, cv_st_dev))

    cv_scores = (cv_mean, cv_st_dev, results['test_score'])
    cv_estimators = results['estimator']

    print(f'\nRunning {k}-model evaluation against Test set...')

    X_test, y_test = load_flatten_test_data(data_path=test_path, target_label='class_label')

    # setup the plot for PR curve
    fig, ax = plt.subplots(figsize=(7, 7))

    count = 1

    eval_results = []
    for est in cv_estimators:

        y_prob = est.predict_proba(X_test)[:, 1]
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        eval_results.append(ap_score)

        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=f'Logistic Regression {count}')

        if save_models:

            # save the model
            filename = f'lr_{count}.sav'
            model_path = output_path
            filepath = os.path.join(model_path, filename)

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            pickle.dump(est, open(filepath, 'wb'))

            count += 1

    ax.legend(loc='upper right')
    ax.set_title('Precision-Recall (PR) curves: Test Evaluation')
    ax.grid(linestyle="--")

    plt.legend()
    plt.show()

    print('Test evaluation results:', eval_results, sep='\n')

    test_mean = np.mean(eval_results) * 100
    test_st_dev = np.std(eval_results) * 100
    print(f'Average Precision Score against Test set: %.2f%% (%.2f%%)' % (test_mean, test_st_dev))

    test_scores = (test_mean, test_st_dev, eval_results)

    print(f'\nRunning {k}-model evaluation against the Environment set...')

    # evaluate against the environment set
    X_test, y_test = load_flatten_env_test_data(data_path=FEAT_PATH, json_base=ENV_FEAT_BASE, target_label='class_label')

    # setup the plot for PR curve
    fig, ax = plt.subplots(figsize=(7, 7))

    count = 1

    env_results = []
    for est in cv_estimators:
        y_prob = est.predict_proba(X_test)[:, 1]
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        env_results.append(ap_score)

        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=f'Logistic Regression {count}')

        count += 1

    ax.legend(loc='upper right')
    ax.set_title('Precision-Recall (PR) curves: Environmental Evaluation')
    ax.grid(linestyle="--")

    plt.legend()
    plt.show()

    print('Environment evaluation results:', env_results, sep='\n')

    env_mean = np.mean(env_results) * 100
    env_st_dev = np.std(env_results) * 100
    print(f'Average Precision Score against Environment set: %.2f%% (%.2f%%)' % (env_mean, env_st_dev))
    env_scores = (env_mean, env_st_dev, env_results)

    return cv_scores, test_scores, env_scores


def train_save_model(train_path=TRAIN_PATH, output_path=OUTPUT_PATH, filename='mfcc_lr.sav', rand_seed=0):
    
    X, y, g = load_flatten_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, solver='liblinear')
    
    # Fit the model on training set
    model.fit(X, y)
    
    # save the model
    filename = os.path.join(output_path, filename)
    pickle.dump(model, open(filename, 'wb'))
    
    print(f'Model saved to: {filename}')


def run_feature_permutation(train_path=TRAIN_PATH, test_path=TEST_PATH, rand_seed=0):

    X, y, g = load_flatten_train_data(data_path=train_path, target_label='class_label')

    X_test, y_test = load_flatten_test_data(data_path=test_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, solver='liblinear')

    model.fit(X, y)

    score = model.score(X_test, y_test)
    print(score, 'Score')

    r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
    print(r.importances_mean)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{i:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


if __name__ == '__main__':
    run_cv()
