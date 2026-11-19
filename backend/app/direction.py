"""Validation-selected auxiliary classifier for next-session return direction."""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import chronological_train_validation_indices


def classifier_input(sequences, flatten):
    sequences = np.asarray(sequences)
    return sequences.reshape(len(sequences), -1) if flatten else sequences[:, -1, :]


def _candidates():
    options = []
    for flatten in (False, True):
        representation = "sequence" if flatten else "latest-day features"
        options.extend([
            (f"{representation} logistic", flatten, LogisticRegression(C=1, max_iter=1000)),
            (f"{representation} gradient boosting", flatten, GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.03, max_depth=1, random_state=0
            )),
            (f"{representation} histogram boosting", flatten, HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.04, max_leaf_nodes=7,
                l2_regularization=2, random_state=0,
            )),
            (f"{representation} random forest", flatten, RandomForestClassifier(
                n_estimators=120, max_depth=4, min_samples_leaf=25,
                max_features="sqrt", random_state=0, n_jobs=-1,
            )),
        ])
    return options


def fit_direction_classifier(train_sequences, train_returns, evaluation_sequences):
    """Select on the chronological training tail, refit before the OOS period."""
    labels = (np.asarray(train_returns) > 0).astype(np.int64)
    train_idx, valid_idx = chronological_train_validation_indices(len(train_sequences), 0.8)
    split = int(train_idx[-1] + 1)
    if split < 2 or len(np.unique(labels[:split])) < 2 or len(np.unique(labels[split:])) < 2:
        raise ValueError("Not enough up and down observations to train the direction classifier")

    choices = []
    for name, flatten, estimator in _candidates():
        x_train = classifier_input(train_sequences[train_idx], flatten)
        x_valid = classifier_input(train_sequences[valid_idx], flatten)
        pipeline = make_pipeline(StandardScaler(), clone(estimator))
        pipeline.fit(x_train, labels[train_idx])
        val_accuracy = float(np.mean(pipeline.predict(x_valid) == labels[valid_idx]))
        choices.append((val_accuracy, name, flatten, estimator))

    val_accuracy, name, flatten, estimator = max(choices, key=lambda item: (item[0], item[1]))
    pipeline = make_pipeline(StandardScaler(), clone(estimator))
    pipeline.fit(classifier_input(train_sequences, flatten), labels)
    eval_x = classifier_input(evaluation_sequences, flatten)
    predictions = pipeline.predict(eval_x).astype(np.int64)
    confidence = pipeline.predict_proba(eval_x).max(axis=1)
    return {
        "model": pipeline,
        "flatten": flatten,
        "name": name,
        "validation_accuracy": val_accuracy,
        "predictions": predictions,
        "confidence": confidence,
    }
