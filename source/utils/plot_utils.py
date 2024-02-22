import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kruskal
from sklearn.metrics import roc_auc_score, roc_curve
from skmisc.loess import loess
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro


def calculate_auc_and_confidence_interval(results):
    """
    Use the R function cvAUC to calculate the AUC and confidence interval.

    Args:
        results (pd.DataFrame): The results of the model with columns "label", "prediction" and "fold".

    Returns:
        dict: A dictionary with the AUC, lower and upper bound of the confidence interval.
    """
    pandas2ri.activate()

    r_df = pandas2ri.py2rpy(results)

    r("library(cvAUC)")
    r(
        """
    calc_auc_ci = function(df) {
        auc = ci.cvAUC(predictions = df$prediction, labels=df$label, folds=df$fold, confidence=0.95)

        return(c(auc$cvAUC, auc$ci))
    }
    """
    )

    auc, lower, upper = r["calc_auc_ci"](r_df)

    return auc, lower, upper


def plot_roc(results, target, ax):
    """
    Plot the ROC curve for the results of a model.

    Args:
        results (pd.DataFrame): The results of the model with columns "prediction", "fold" and one named after the target variable.
        target (str): The name of the target variable.
        ax (matplotlib.axes.Axes): The axes to plot the ROC curve on.
    """

    results["label"] = results[target]

    auc, lower, upper = calculate_auc_and_confidence_interval(results)

    tprs = []
    aucs = []
    n = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_ix in results.fold.unique():
        subset = results[results.fold == fold_ix]
        fold_auc = roc_auc_score(subset[target], subset["prediction"])
        aucs.append(fold_auc)
        fpr, tpr, _ = roc_curve(subset[target], subset["prediction"])
        ax.plot(
            fpr,
            tpr,
            c="gray",
            lw=1,  # , label=f"Fold {fold_ix} -- AUC = {fold_auc:.2f}"
        )

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs.append(interp_tpr)

        n.append(len(subset))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1

    ax.plot([0, 1], [0, 1], linestyle="--", c="orange")

    ax.set_title(f"ROC curve for predicting {target}")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")

    ax.plot(
        mean_fpr,
        mean_tpr,
        c="b",
        lw=3,
        label=f"AUC = {auc:.2f} [95% CI {lower:.2f}-{upper:.2f}]",
    )

    ax.legend(loc="lower right")


def plot_calibration_curve(results, target, ax):
    loess_regression = loess(results["prediction"], results[target])
    loess_regression.fit()
    prediction = loess_regression.predict(
        results["prediction"].sort_values(), stderror=True
    )

    ax.set_title(f"Calibration curve for predicting {target}")
    ax.set_xlabel(f"Predicted probability of {target}")
    ax.set_ylabel(f"True probability of {target}")
    ax.fill_between(
        results["prediction"].sort_values(),
        np.clip(prediction.values - prediction.stderr, 0, 1),
        np.clip(prediction.values + prediction.stderr, 0, 1),
        alpha=0.3,
    )
    ax.plot(results["prediction"].sort_values(), np.clip(prediction.values, 0, 1))
    ax.plot([0, 1], [0, 1], linestyle="--", c="gray")

    bins = np.linspace(0, 1, 61)

    neg_heights, neg_bins = np.histogram(
        results[results[target] == 0]["prediction"], bins=bins
    )
    pos_heights, pos_bins = np.histogram(
        results[results[target] == 1]["prediction"], bins=bins
    )
    ax.bar(
        pos_bins[:-1] + (neg_bins[1] - neg_bins[0]) / 2,
        -neg_heights / max(neg_heights.max(), pos_heights.max()) / 10,
        width=0.01,
        color="C1",
    )
    ax.bar(
        pos_bins[:-1] + (pos_bins[1] - pos_bins[0]) / 2,
        pos_heights / max(neg_heights.max(), pos_heights.max()) / 10,
        width=0.01,
        color="C0",
    )
    ax.text(0.85, 0.02, "1", fontsize=12, c="C0", weight="bold")
    ax.text(0.85, -0.04, "0", fontsize=12, c="C1", weight="bold")


def plot_correlation(a, b, data, ax):
    r, p = pearsonr(data[a], data[b])

    ax.set_title(f"Pearson's r = {r:.3f}, p = {p:.3f}")
    ax.scatter(data[a], data[b], alpha=0.5)
    ax.legend()
    ax.set_xlabel(a)
    ax.set_ylabel(b)


def plot_boxplot(predictor, group, data, ax, order=None):
    _, p = kruskal(
        *[
            data[data[group] == value][predictor].values
            for value in data[group].unique()
        ]
    )

    ax.set_title(f"Kruskal-Wallis p = {p:.3f}")
    sn.boxenplot(x=group, y=predictor, data=data, ax=ax, order=order)
