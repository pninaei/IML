from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        best_threshold, best_feature, best_sign, best_error = None, None, None, float('inf')
        
        for feature, sign in product(range(n_features), [-1, 1]):
            threshold, error = self._find_threshold(X[:, feature], y, sign)
            if error < best_error:
                best_threshold, best_feature, best_sign, best_error = threshold, feature, sign, error
        
        self.threshold_, self.j_, self.sign_ = best_threshold, best_feature, best_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort the values and corresponding labels
        ids = np.argsort(values)
        values = values[ids]
        labels = labels[ids]

        # Calculate the initial loss for classifying all as `sign`
        # This represents the loss if the threshold is smaller than the smallest value
        initial_loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        # Calculate the cumulative sum of labels multiplied by `sign`
        cumulative_sum = np.cumsum(labels * sign)

        # Adjust the initial loss for each possible threshold
        loss = initial_loss - cumulative_sum

        # Append the initial loss to the beginning of the loss array
        loss = np.append(initial_loss, loss)

        # Find the index of the minimum loss
        min_loss_index = np.argmin(loss)

        # Define the threshold values, including -inf and inf
        thresholds = np.concatenate([[-np.inf], values[1:], [np.inf]])

        # Get the threshold corresponding to the minimum loss
        best_threshold = thresholds[min_loss_index]

        # Get the minimum loss value
        min_loss = loss[min_loss_index]

        # Return the best threshold and the corresponding minimum loss
        return best_threshold, min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        
        return misclassification_error(y, self._predict(X))
