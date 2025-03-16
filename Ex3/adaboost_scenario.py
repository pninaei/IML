import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size,
                                                                                           noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    T = np.arange(1, n_learners + 1)
    train_errors = np.zeros(n_learners)
    test_errors = np.zeros(n_learners)

    for t in T:
        train_errors[t - 1] = adaboost.partial_loss(train_X, train_y, t)
        test_errors[t - 1] = adaboost.partial_loss(test_X, test_y, t)

    # Create a figure with Plotly
    fig = go.Figure()

    # Add traces for training and test errors
    fig.add_trace(go.Scatter(x=T, y=train_errors, mode='lines', name='Training Error'))
    fig.add_trace(go.Scatter(x=T, y=test_errors, mode='lines', name='Test Error'))

    # Update layout
    fig.update_layout(
        title=f'Training and Test Errors of AdaBoost as function of number of classifiers (Noise={noise})',
        xaxis_title='Iterations',
        yaxis_title='Misclassification Error',
        legend=dict(x=1.05, y=1.0),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(f"adaboost error with {noise}.html")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array(
        [np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"Iteration {t}" for t in T])

    for i, t in enumerate(T):
        # Predict the labels for the decision surface
        predict = lambda X: adaboost.partial_predict(X, t)
        decision_plot = decision_surface(predict, lims[0], lims[1], density=120, showscale=False)

        # Add decision boundary plot to subplot
        fig.add_trace(decision_plot, row=1, col=i + 1)

        # Add test data scatter plot
        test_scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                                  marker=dict(color=test_y, symbol=class_symbols[test_y.astype(int)],
                                              colorscale=custom, line=dict(width=1, color='black')),
                                  name='Test Data')

        fig.add_trace(test_scatter, row=1, col=i + 1)
    fig.write_html(f"adaboost_decision_boundaries with {noise}.html")

    # Question 3: Decision surface of best performing ensemble
    # Find the best performing ensemble size
    best_t = np.argmin(test_errors) + 1
    min_error = test_errors[best_t - 1]
    accuracy = 1 - min_error

    # Plot decision surface of best performing ensemble
    predict = lambda X: adaboost.partial_predict(X, best_t)
    fig_best = make_subplots(rows=1, cols=1,
                             subplot_titles=[f"Best Ensemble (Size={best_t}, Accuracy={accuracy:.2f})"])

    decision_plot = decision_surface(predict, lims[0], lims[1], density=120, showscale=False)
    fig_best.add_trace(decision_plot, row=1, col=1)

    test_scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                              marker=dict(color=test_y, symbol=class_symbols[(test_y+1).astype(int)//2],
                                          colorscale=custom, line=dict(width=1, color='black')),name='Test Data')

    fig_best.add_trace(test_scatter, row=1, col=1)
    fig_best.write_html(f"adaboost_best_decision_boundary {noise}.html")

    # Question 4: Decision surface with weighted samples
    # Plot the training set with point size proportional to weight
    weights = (adaboost.D_ / np.max(adaboost.D_)) * 5
    fig = go.Figure([
        decision_surface(lambda X: adaboost.partial_predict(X, n_learners), lims[0,:], lims[1,:], density=60, showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(size=weights, color=train_y, symbol=class_symbols[(train_y+1).astype(int)]))],
        layout=go.Layout(width=500, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
                         title=f"Final AdaBoost Sample Distribution"))
    fig.write_html(f"adaboost_{noise}_weighted_samples.html")


if __name__ == '__main__':
    np.random.seed(0)
    for i in [0, 0.4]:
        fit_and_evaluate_adaboost(i)


