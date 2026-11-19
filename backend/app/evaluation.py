"""Leakage-conscious metrics and simple, execution-delayed strategy diagnostics."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chronological_train_validation_indices(n_observations, train_fraction=0.8):
    """Contiguous split indices; never shuffles temporal observations."""
    if n_observations < 2 or not 0 < train_fraction < 1:
        raise ValueError("Need at least two observations and a train fraction between 0 and 1")
    split = int(n_observations * train_fraction)
    if split < 1 or split >= n_observations:
        raise ValueError("Chronological split leaves an empty partition")
    return np.arange(split), np.arange(split, n_observations)


def fit_minmax_scaler(training_sequences):
    """Fit feature bounds on training samples only, across their time/feature axes."""
    values = np.asarray(training_sequences, dtype=float)
    if values.ndim != 3 or not len(values):
        raise ValueError("Expected non-empty [samples, time, features] training data")
    x_min = values.reshape(-1, values.shape[-1]).min(axis=0)
    x_max = values.reshape(-1, values.shape[-1]).max(axis=0)
    x_max = np.where(x_max == x_min, x_min + 1e-6, x_max)
    return x_min, x_max


def apply_minmax_scaler(sequences, x_min, x_max):
    values = np.asarray(sequences, dtype=float)
    return (values - x_min) / (x_max - x_min)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def wilson_interval(successes, total, z=1.959963984540054):
    """Wilson score interval for a binomial proportion, returned as percentages."""
    if total <= 0:
        return [None, None]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [round(100 * max(0.0, center - radius), 2), round(100 * min(1.0, center + radius), 2)]


def direction_metrics(y_true, y_pred):
    """Metrics for binary labels encoded as down=0, up=1."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    counts = np.bincount(y_true, minlength=2)
    return {
        "n": n,
        "accuracy": round(100 * accuracy_score(y_true, y_pred), 2) if n else None,
        "accuracy_ci95": wilson_interval(correct, n),
        "balanced_accuracy": round(100 * balanced_accuracy_score(y_true, y_pred), 2) if n else None,
        "precision_up": round(100 * precision_score(y_true, y_pred, zero_division=0), 2) if n else None,
        "recall_up": round(100 * recall_score(y_true, y_pred, zero_division=0), 2) if n else None,
        "f1_up": round(100 * f1_score(y_true, y_pred, zero_division=0), 2) if n else None,
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4) if n else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "class_distribution": {"down": int(counts[0]), "up": int(counts[1])},
    }


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1 - np.sum(errors**2) / ss_tot) if ss_tot > 0 else None,
    }


def confidence_buckets(y_true, y_pred, probabilities):
    """Equal-count confidence bins; report empirical accuracy, not calibrated odds."""
    if probabilities is None:
        return []
    confidence = np.asarray(probabilities, dtype=float)
    if len(confidence) != len(y_true) or not len(confidence):
        return []
    groups = np.array_split(np.argsort(confidence, kind="stable"), min(3, len(confidence)))
    labels = ["low", "medium", "high"][-len(groups):]
    return [
        {
            "bucket": label,
            "n": int(len(idx)),
            "mean_confidence": round(float(confidence[idx].mean()) * 100, 2),
            "accuracy": round(float(np.mean(np.asarray(y_true)[idx] == np.asarray(y_pred)[idx])) * 100, 2),
        }
        for label, idx in zip(labels, groups) if len(idx)
    ]


def backtest_long_flat(actual_log_returns, predicted_up, cost_bps=10):
    """Conservative close-only diagnostic: signal is executed one session later.

    Positions are long after an up signal and flat after a down signal. The one-row
    lag avoids assuming an execution at the same close used to form the signal.
    Costs are charged on absolute position changes (entry/exit turnover).
    """
    log_returns = np.asarray(actual_log_returns, dtype=float)
    signal = np.asarray(predicted_up, dtype=int).astype(float)
    position = np.zeros(len(signal), dtype=float)
    if len(position) > 1:
        position[1:] = signal[:-1]
    asset_returns = np.expm1(log_returns)
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    strategy_returns = position * asset_returns - turnover * (cost_bps / 10000)
    wealth = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / peak - 1 if len(wealth) else np.array([])
    active = strategy_returns[position > 0]
    losses = abs(float(strategy_returns[strategy_returns < 0].sum()))
    std = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = float(np.mean(strategy_returns) / std * np.sqrt(252)) if std > 0 else None
    bh = np.prod(1 + asset_returns) - 1 if len(asset_returns) else 0.0
    return {
        "execution": "one-session delayed close-only, long/flat",
        "cost_bps_per_turnover_unit": cost_bps,
        "cumulative_return_pct": round((float(wealth[-1]) - 1) * 100, 2) if len(wealth) else 0.0,
        "buy_and_hold_return_pct": round(float(bh) * 100, 2),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2) if len(drawdown) else 0.0,
        "win_rate_pct": round(float(np.mean(active > 0)) * 100, 2) if len(active) else None,
        "profit_factor": round(float(strategy_returns[strategy_returns > 0].sum()) / losses, 3) if losses else None,
        "turnover_units": round(float(turnover.sum()), 2),
        "observations": int(len(log_returns)),
    }


def _classification_row(y_true, y_pred, name, probabilities=None):
    return {
        "model": name,
        **direction_metrics(y_true, y_pred),
        "confidence_buckets": confidence_buckets(y_true, y_pred, probabilities),
    }


def evaluation_suite(
    train_sequences,
    train_returns,
    test_sequences,
    test_returns,
    prior_close,
    actual_close,
    moving_average_close,
    transformer_raw_returns,
    selected_direction,
    train_volatility,
    test_volatility,
):
    """Fit train-only reference models and report identical chronological test rows."""
    y_train = np.asarray(train_returns, dtype=float)
    y_test = np.asarray(test_returns, dtype=float)
    y = (y_test > 0).astype(int)
    xtr = np.asarray(train_sequences)[:, -1, :]
    xte = np.asarray(test_sequences)[:, -1, :]

    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=0))
    logistic.fit(xtr, (y_train > 0).astype(int))
    logistic_pred = logistic.predict(xte).astype(int)
    logistic_prob = logistic.predict_proba(xte).max(axis=1)
    selected_pred = np.asarray(selected_direction["predictions"], dtype=int)
    raw_transformer_sign = (np.asarray(transformer_raw_returns) > 0).astype(int)

    always_up = np.ones(len(y), dtype=int)
    majority_label = int(np.mean(y_train > 0) >= 0.5)
    majority = np.full(len(y), majority_label, dtype=int)

    # These rolling mean prices use only observations through each forecast origin.
    close_models = {
        "Persistence · previous close": np.asarray(prior_close),
        "Moving average · previous 5 closes": np.asarray(moving_average_close),
    }

    # Fit simple return regressors on all pre-test data; preprocessors live inside each pipeline.
    linear_log = make_pipeline(StandardScaler(), LinearRegression())
    linear_log.fit(xtr, y_train)
    linear_log_return = linear_log.predict(xte)
    close_models["Linear regression · log return"] = np.asarray(prior_close) * np.exp(linear_log_return)

    linear_pct = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    linear_pct.fit(xtr, np.expm1(y_train))
    pct_pred = linear_pct.predict(xte)
    close_models["Ridge · percentage return"] = np.asarray(prior_close) * (1 + pct_pred)

    vol_scale_train = np.maximum(np.asarray(train_volatility), 1e-6)
    vol_scale_test = np.maximum(np.asarray(test_volatility), 1e-6)
    linear_vol = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    linear_vol.fit(xtr, y_train / vol_scale_train)
    normalized_return = linear_vol.predict(xte) * vol_scale_test
    close_models["Ridge · volatility-normalized log return"] = np.asarray(prior_close) * np.exp(normalized_return)

    predictions = {
        "Always up": always_up,
        "Training majority class": majority,
        "Logistic regression · log-return direction": logistic_pred,
        "Transformer raw-return sign": raw_transformer_sign,
    }
    probabilities = {"Logistic regression · log-return direction": logistic_prob}
    selected_name = selected_direction.get("model", "Transformer raw-return sign")
    if selected_name != "Transformer raw-return sign":
        predictions[selected_name] = selected_pred
        if selected_direction.get("confidence") is not None:
            probabilities[selected_name] = np.asarray(selected_direction["confidence"], dtype=float)

    # Neutral threshold is a predeclared causal rule: 25% of trailing 20-session
    # volatility, with a 10bp floor. It is not fitted on test outcomes.
    tr_threshold = np.maximum(0.001, 0.25 * vol_scale_train)
    te_threshold = np.maximum(0.001, 0.25 * vol_scale_test)
    y3_train = np.where(y_train > tr_threshold, 2, np.where(y_train < -tr_threshold, 0, 1))
    y3_test = np.where(y_test > te_threshold, 2, np.where(y_test < -te_threshold, 0, 1))
    three_class = RandomForestClassifier(
        n_estimators=160, max_depth=5, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=0, n_jobs=-1,
    )
    three_class.fit(xtr, y3_train)
    y3_pred = three_class.predict(xte).astype(int)
    three_class_report = {
        "model": "Random forest · three-class direction",
        "target": "down / neutral / up; neutral = max(10bp, 0.25 × trailing 20-session volatility)",
        "n": int(len(y3_test)),
        "accuracy": round(float(np.mean(y3_test == y3_pred)) * 100, 2),
        "macro_f1": round(float(f1_score(y3_test, y3_pred, labels=[0, 1, 2], average="macro", zero_division=0)) * 100, 2),
        "confusion_matrix": confusion_matrix(y3_test, y3_pred, labels=[0, 1, 2]).tolist(),
        "class_distribution": {"down": int(np.sum(y3_test == 0)), "neutral": int(np.sum(y3_test == 1)), "up": int(np.sum(y3_test == 2))},
        "predicted_class_distribution": {"down": int(np.sum(y3_pred == 0)), "neutral": int(np.sum(y3_pred == 1)), "up": int(np.sum(y3_pred == 2))},
    }

    # Expanding, one-step logistic refit. At each date it sees labels strictly before
    # that target date, so prior test observations enter only after they have occurred.
    walkforward_pred = np.zeros(len(y), dtype=int)
    walkforward_prob = np.zeros(len(y), dtype=float)
    train_n = len(y_train)
    full_x = np.concatenate([xtr, xte], axis=0)
    full_y = np.concatenate([(y_train > 0).astype(int), y], axis=0)
    for i in range(len(y)):
        fit_end = train_n + i
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=0))
        estimator.fit(full_x[:fit_end], full_y[:fit_end])
        walkforward_pred[i] = int(estimator.predict(xte[i:i + 1])[0])
        walkforward_prob[i] = float(estimator.predict_proba(xte[i:i + 1]).max())
    predictions["Expanding logistic · refit each session"] = walkforward_pred
    probabilities["Expanding logistic · refit each session"] = walkforward_prob

    direction_rows = [
        _classification_row(y, pred, name, probabilities.get(name))
        for name, pred in predictions.items()
    ]

    notebook_predicted_close = np.asarray(prior_close) * np.exp(np.asarray(transformer_raw_returns))
    close_models["Transformer · notebook model"] = notebook_predicted_close
    if selected_name != "Transformer raw-return sign":
        selected_direction_close = np.asarray(prior_close) * np.exp(
            np.where(selected_pred == 1, 1.0, -1.0) * np.abs(np.asarray(transformer_raw_returns))
        )
        close_models["Validation-selected direction + Transformer magnitude"] = selected_direction_close
    regression_rows = [
        {"model": name, **regression_metrics(actual_close, pred)}
        for name, pred in close_models.items()
    ]

    selected_row = next(row for row in direction_rows if row["model"] == selected_name)
    always_up_accuracy = next(row["accuracy"] for row in direction_rows if row["model"] == "Always up")
    majority_accuracy = next(row["accuracy"] for row in direction_rows if row["model"] == "Training majority class")
    backtest = backtest_long_flat(y_test, selected_pred, cost_bps=10)
    return {
        "target": "next-session close-to-close log return; binary up iff return > 0",
        "horizon": "1 trading session",
        "regression_split": "2025-01-01 onward; chronological holdout",
        "direction_split": "2025-01-01 onward; chronological holdout",
        "directional_baseline_accuracy": max(always_up_accuracy, majority_accuracy),
        "training_majority_class_accuracy": majority_accuracy,
        "regression_models": regression_rows,
        "direction_models": direction_rows,
        "three_class_experiment": three_class_report,
        "trading_diagnostic": backtest,
        "selected_direction": selected_row,
    }
