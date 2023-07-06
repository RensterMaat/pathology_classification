import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kruskal
from sklearn.metrics import roc_auc_score, roc_curve
from skmisc.loess import loess


def plot_roc(predictor, target, data, ax, aggregated_auc=None):
    tprs = []
    aucs = []
    n = []
    mean_fpr = np.linspace(0, 1, 100)

    for center in data.center.unique():
        subset = data[data.center == center]
        auc = roc_auc_score(subset[target], subset[predictor])
        aucs.append(auc)
        fpr, tpr, _ = roc_curve(subset[target], subset[predictor])
        if aggregated_auc is None:
            ax.plot(fpr, tpr, c="gray", lw=1, label=f"{center} -- {auc:.3f}")
        else:
            ax.plot(fpr, tpr, c="gray", lw=1)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs.append(interp_tpr)

        n.append(len(subset))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1

    ax.plot([0, 1], [0, 1], linestyle="--", c="orange")

    ax.set_title(f"ROC curve - {predictor} to predict {target}")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")

    if aggregated_auc is None:
        ax.plot(
            mean_fpr,
            mean_tpr,
            c="b",
            lw=3,
            label=f"Mean -- {np.average(aucs, weights=n):.3f}",
        )
    else:
        ax.plot(
            mean_fpr,
            mean_tpr,
            c="b",
            lw=3,
            label=f"AUC = {aggregated_auc[0]:.3f} [95% CI {aggregated_auc[1]:.3f}-{aggregated_auc[2]:.3f}]",
        )

    ax.legend(loc="lower right")


def plot_calibration_curve(predictor, target, data, ax):
    loess_regression = loess(data[predictor], data[target])
    loess_regression.fit()
    prediction = loess_regression.predict(data[predictor].sort_values(), stderror=True)

    ax.set_xlabel("Predicted probability of benefit")
    ax.set_ylabel("True probability of benefit")
    ax.fill_between(
        data[predictor].sort_values(),
        np.clip(prediction.values - prediction.stderr, 0, 1),
        np.clip(prediction.values + prediction.stderr, 0, 1),
        alpha=0.3,
    )
    ax.plot(data[predictor].sort_values(), np.clip(prediction.values, 0, 1))
    ax.plot([0, 1], [0, 1], linestyle="--", c="gray")

    bins = np.linspace(0, 1, 61)

    neg_heights, neg_bins = np.histogram(data[data[target] == 0][predictor], bins=bins)
    pos_heights, pos_bins = np.histogram(data[data[target] == 1][predictor], bins=bins)
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
