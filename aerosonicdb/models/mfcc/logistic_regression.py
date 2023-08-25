import os
import json
import pickle
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from aerosonicdb.utils import get_project_root
from aerosonicdb.utils import fetch_k_fold_cv_indicies
from aerosonicdb.utils import load_frame_train_data
from aerosonicdb.utils import load_frame_test_data
from aerosonicdb.utils import load_frame_env_test_data

root_path = get_project_root()
feat_path = os.path.join(root_path, 'data/processed')
train_path = os.path.join(feat_path, '13_mfcc_5_train.json')
test_path = os.path.join(feat_path, '13_mfcc_5_test.json')
env_feat_base = '_ENV_13_mfcc_5.json'
output_path = os.path.join(root_path, 'models')

if not os.path.exists(output_path):
    os.makedirs(output_path)


def run_cv(train_path=train_path, k=10, rand_seed=0, test_path=test_path):
    X, y, g = load_frame_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)
    print(f'Running {k}-fold cross-validation...')
    results = cross_validate(model, X, y,
                             cv=fetch_k_fold_cv_indicies(X, y, g, k=k),
                             scoring='average_precision',
                             n_jobs=-1,
                             return_estimator=True)

    print('CV results:', results['test_score'], sep='\n')

    cv_mean = results['test_score'].mean() * 100
    cv_st_dev = results['test_score'].std() * 100

    print(f'Average Precision Score for 10-fold CV: %.2f%% (%.2f%%)' % (cv_mean, cv_st_dev))

    cv_scores = (cv_mean, cv_st_dev, results['test_score'])

    cv_estimators = results['estimator']
    print(f'\nRunning {k}-model evaluation against Test set...')
    X_test, y_test = load_frame_test_data(data_path=test_path, target_label='class_label')
    eval_results = []
    for est in cv_estimators:
        y_prob = est.predict_proba(X_test)[:, 1]
        # print(y_test[:5])
        # print(y_prob[:5])
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        eval_results.append(ap_score)

    print('Test evaluation results:', eval_results, sep='\n')

    test_mean = np.mean(eval_results) * 100
    test_st_dev = np.std(eval_results) * 100
    print(f'Average Precision Score against Test set: %.2f%% (%.2f%%)' % (test_mean, test_st_dev))

    test_scores = (test_mean, test_st_dev, eval_results)

    print(f'\nRunning {k}-model evaluation against the Environment set...')

    # evaluate against the environment set
    X_test, y_test = load_frame_env_test_data(data_path=feat_path, json_base=env_feat_base, target_label='class_label')

    env_results = []
    for est in cv_estimators:
        y_prob = est.predict_proba(X_test)[:, 1]
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


def train_save_model(train_path=train_path, output_path=output_path, filename='mfcc_logistic_regression.sav', rand_seed=0):
    
    X, y, g = load_frame_train_data(data_path=train_path, target_label='class_label')

    model = LogisticRegression(random_state=rand_seed, max_iter=400)
    
    # Fit the model on training set
    model.fit(X, y)
    
    # save the model
    filename = os.path.join(output_path, filename)
    pickle.dump(model, open(filename, 'wb'))
    
    print(f'Model saved to: {filename}')


if __name__ == '__main__':
    run_cv()
