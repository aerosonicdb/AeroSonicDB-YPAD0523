import os

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.utils import class_weight

from aerosonicdb.utils import (
    fetch_k_fold_cv_indicies,
    get_project_root,
    load_env_test_data,
    load_test_data,
    load_train_data,
    plot_history,
    train_val_split,
)

absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


ROOT_PATH = get_project_root()
FEAT_PATH = os.path.join(ROOT_PATH, "data/processed")
TRAIN_PATH = os.path.join(FEAT_PATH, "13_mfcc_5_train.json")
TEST_PATH = os.path.join(FEAT_PATH, "13_mfcc_5_test.json")
ENV_FEAT_BASE = "_ENV_13_mfcc_5.json"
OUTPUT_PATH = os.path.join(ROOT_PATH, "models")


def build_model(x):
    model = keras.Sequential(
        [
            # input layer
            keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),
            # 1st dense layer
            keras.layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.4),
            # 2nd dense layer
            keras.layers.Dense(
                32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.4),
            # output layer
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimiser,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="PR-AUC")],
    )

    return model


def run_cv(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    output_path=OUTPUT_PATH,
    epochs=1,
    batch_size=216,
    rand_seed=0,
    verbose=0,
    k=5,
    save_models=True,
):

    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_label="class_label")
    build = build_model(X)

    model = KerasClassifier(
        model=build,
        epochs=epochs,
        batch_size=batch_size,
        random_state=rand_seed,
        verbose=verbose,
        class_weight="balanced",
    )

    print(f"Running {k}-fold cross-validation...")

    results = cross_validate(
        model,
        X,
        y,
        cv=fetch_k_fold_cv_indicies(X, y, g, k=k),
        scoring="average_precision",
        return_estimator=True,
    )

    print("CV results:", results["test_score"], sep="\n")

    cv_mean = results["test_score"].mean() * 100
    cv_st_dev = results["test_score"].std() * 100

    print(
        f"Average Precision Score for 5-fold CV: %.2f%% (%.2f%%)" % (cv_mean, cv_st_dev)
    )

    cv_scores = (cv_mean, cv_st_dev, results["test_score"])
    cv_estimators = results["estimator"]

    print(f"\nRunning {k}-model evaluation against Test set...")

    X_test, y_test = load_test_data(data_path=test_path, target_label="class_label")

    # setup the plot for PR curve
    fig, ax = plt.subplots(figsize=(5, 4))

    count = 1

    eval_results = []
    for est in cv_estimators:

        y_prob = est.predict_proba(X_test, batch_size=batch_size)[:, 1]
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        eval_results.append(ap_score)

        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob, ax=ax, name=f"MLP {count}"
        )

        if save_models:
            # save the model
            model_path = os.path.join(output_path, f"mlp_{count}", "model")

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            est.model_.save(model_path)
            count += 1

    ax.legend(loc="upper right")
    ax.set_title("MLP PR curves: TEST")
    ax.grid(linestyle="--")

    plt.legend()

    # check for/create output directory for figures
    if not os.path.isdir("../figures"):
        os.mkdir("../figures")

    plt.savefig(f"../figures/MLP_Test_PR_curves.png", dpi=300)
    plt.show()

    print("Test evaluation results:", eval_results, sep="\n")

    test_mean = np.mean(eval_results) * 100
    test_st_dev = np.std(eval_results) * 100
    print(
        f"Average Precision Score against Test set: %.2f%% (%.2f%%)"
        % (test_mean, test_st_dev)
    )

    test_scores = (test_mean, test_st_dev, eval_results)

    print(f"\nRunning {k}-model evaluation against the Environment set...")
    # evaluate against the environment set
    X_test, y_test = load_env_test_data(
        data_path=FEAT_PATH, json_base=ENV_FEAT_BASE, target_label="class_label"
    )

    # setup the plot for PR curve
    fig, ax = plt.subplots(figsize=(5, 4))

    count = 1
    env_results = []
    for est in cv_estimators:
        y_prob = est.predict_proba(X_test, batch_size=batch_size)[:, 1]
        ap_score = average_precision_score(y_true=y_test, y_score=y_prob)
        env_results.append(ap_score)

        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob, ax=ax, name=f"MLP {count}"
        )

        count += 1

    ax.legend(loc="upper right")
    ax.set_title("MLP PR curves: ENV")
    ax.grid(linestyle="--")

    plt.legend()
    plt.savefig(f"../figures/MLP_Env_PR_curves.png", dpi=300)
    plt.show()

    print("Environment evaluation results:", env_results, sep="\n")

    env_mean = np.mean(env_results) * 100
    env_st_dev = np.std(env_results) * 100
    print(
        f"Average Precision Score against Environment set: %.2f%% (%.2f%%)"
        % (env_mean, env_st_dev)
    )
    env_scores = (env_mean, env_st_dev, env_results)

    return cv_scores, test_scores, env_scores


def train_save_model(
    output_path=OUTPUT_PATH,
    train_path=TRAIN_PATH,
    filename="mfcc_mlp_n",
    epochs=1,
    batch_size=216,
    verbose=0,
    rand_seed=0,
):

    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_label="class_label")

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    # X_train = np.concatenate((X_train, X_val))
    # y_train = np.concatenate((y_train, y_val))

    model = build_model(X)
    model.summary()

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )

    print(class_weights[:5])

    class_weight_dict = dict(enumerate(class_weights))

    print(class_weight_dict)

    # train model
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        class_weight=class_weight_dict,
    )

    # save the model
    model_path = os.path.join(output_path, filename, "model")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(model_path)

    print(f"Model saved to {model_path}.\n")


def train_plot_model(
    train_path=TRAIN_PATH, epochs=50, batch_size=216, verbose=1, rand_seed=0
):

    keras.utils.set_random_seed(rand_seed)

    X, y, g = load_train_data(data_path=train_path, target_label="class_label")

    X_train, y_train, X_val, y_val = train_val_split(X, y, g)

    model = build_model(X)
    model.summary()

    # train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    # save the model
    plot_history(history)


if __name__ == "__main__":
    run_cv(epochs=5, save_models=False)
    # train_plot_model()
    # train_save_model(epochs=50)
