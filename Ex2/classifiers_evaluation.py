

from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def record_loss(perceptron, losses, X, y):

    losses.append(perceptron.loss(X, y))


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        callback = lambda per, _, __: record_loss(per, losses, X, y)
        Perceptron(callback=callback).fit(X, y)

        # # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=np.arange(len(losses)), y=losses, mode="lines", 
                marker_color="blue"), layout=go.Layout(title=f"Perceptron Training Loss - {n} Dataset"
                                                       , xaxis_title="Iteration", yaxis_title="Loss"))
        
        fig.write_html(f"perceptron_{n}.html")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(X, y)
        lda = LDA().fit(X, y)

        gnb_pred = gnb.predict(X)
        lda_pred = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from loss_functions import accuracy
        gnb_accuracy = accuracy(y, gnb_pred)
        lda_accuracy = accuracy(y, lda_pred)

        # Create the subplot
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Gaussian Naive Bayes (accuracy={round(gnb_accuracy, 2)})",
                                            f"LDA (accuracy={round(lda_accuracy,2)})"))

        fig.update_layout(title_text=f'Gaussian Naive Bayes vs LDA estimators comparison over {f.split(".")[0]}',
                          font=dict(size=15),title_pad=dict(b=20))
        # Increase font size of subplot titles
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=20)  # Adjust the font size as desired
            annotation['y'] -= 0.05
        # Add traces for data points, setting symbols and colors

        unique_classes = np.unique(y)
        colors = class_colors(len(unique_classes))

        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                   marker=dict(color=[colors[label][1] for label in gnb_pred],
                                               symbol=[class_symbols[label] for label in y],
                                               )),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                   marker=dict(color=[colors[label][1] for label in gnb_pred],
                                               symbol=[class_symbols[label] for label in y],
                                               ))],
                       rows=[1, 1], cols=[1, 2])

        # Add 'X' dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x", size=8)),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x", size=8))],
                       rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(unique_classes)):
            fig.add_traces([get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])),
                            get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.write_html(f"LDA_vs_naive_bayes_{f}.html")
        
if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
