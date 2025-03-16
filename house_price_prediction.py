import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X = X.drop_duplicates()  # drop duplicates
    y = y.loc[X.index]

    X = X.drop(["id", "date", "lat", "long"], axis=1)  # drop id and date columns

    # Drop rows with any missing values in x and y
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    # Remove samples with invalid data
    valid_indices = X[X['yr_built'] <= X['yr_renovated']].index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    valid_data = (X["waterfront"].isin([0, 1]) & X["view"].isin(range(5)) & X["condition"].isin(
        range(1, 6)) & X["grade"].isin(range(1, 14)))

    X = X.loc[valid_data]
    y = y.loc[valid_data]

    for column in ["sqft_living", "sqft_lot", "yr_built", "floors", "sqft_living15", "sqft_lot15"]:
        valid = X[column] > 0
        X, y = X[valid], y[valid]

    for column in ["sqft_basement", "yr_renovated", "sqft_above", "bathrooms", "bedrooms"]:
        valid = X[column] >= 0
        X, y = X[valid], y[valid]

        # Create new features
    X['yrs_since_renovation'] = X['yr_renovated'] - X['yr_built']

    # Handle cases where there was no renovation
    X['yrs_since_renovation'] = X['yrs_since_renovation'].where(X['yr_renovated'] != 0, 0)

    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = X.drop(["id", "lat", "long", "date"], axis=1)
    # Create new features
    X['yrs_since_renovation'] = X['yr_renovated'] - X['yr_built']
    X['yrs_since_renovation'] = X['yrs_since_renovation'].fillna(0)

    # Handle cases where there was no renovation
    X['yrs_since_renovation'] = X['yrs_since_renovation'].where(X['yr_renovated'] != 0, 0)

    medians_dict = X.median().to_dict()

    # Replace invalid feature values with median values
    X["waterfront"] = X["waterfront"].apply(lambda x: x if x in [0, 1] else medians_dict["waterfront"])
    X["view"] = X["view"].apply(lambda x: x if x in range(5) else medians_dict["view"])
    X["condition"] = X["condition"].apply(lambda x: x if x in range(1, 6) else medians_dict["condition"])
    X["grade"] = X["grade"].apply(lambda x: x if x in range(1, 14) else medians_dict["grade"])

    for column in ["sqft_living", "sqft_lot", "yr_built", "floors", "sqft_living15", "sqft_lot15"]:
        X[column] = X[column].apply(lambda x: x if x > 0 else medians_dict[column])

    for column in ["sqft_basement", "yr_renovated", "sqft_above", "bathrooms", "bedrooms"]:
        X[column] = X[column].apply(lambda x: x if x >= 0 else medians_dict[column])

    X = X.fillna(0)

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "."):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)  # Standard deviation of the response
    for column in X.columns:
        feature = X[column]
        cov_xy = np.cov(feature, y)[0, 1]  # Covariance between feature and response
        std_x = np.std(feature)  # Standard deviation of the feature

        pearson_correlation = cov_xy / (std_x * std_y)

        plt.figure()
        plt.scatter(feature, y, alpha=0.5)

        plt.title(f'{column} vs Response\nPearson Correlation: {pearson_correlation}')
        plt.xlabel(column)
        plt.ylabel('Response')

        # Save plot:
        plot_path = f"{output_path}/{column}_vs_Response.png"
        plt.savefig(plot_path)
        plt.close()


def fitting(X_train, y_train, X_test, y_test):
    """Fit linear model over training data and evaluate over test data.
    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Training data to fit an estimator for

    y_train : ndarray of shape (n_samples, )
        Training responses to fit to

    X_test : ndarray of shape (m_samples, n_features)
        Test data to evaluate the estimator

    y_test : ndarray of shape (m_samples, )
        Test responses to evaluate the estimator

    Returns
    -------
    The loss of the fitted model over the test set
    """
    model = LinearRegression()
    percentages = np.arange(10, 101, 1)
    mean_losses = []
    std_losses = []

    for percents in percentages:
        losses = []
        for _ in range(10):
            # Sample p% of the overall training data use pandas.DataFrame.sample
            X_sample = X_train.sample(frac=percents / 100, random_state=np.random.randint(1, 10000))
            y_sample = y_train.loc[X_sample.index]

            # fit linear model
            model.fit(X_sample, y_sample)
            # Calculate loss over test set
            loss = model.loss(X_test, y_test)
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    return mean_losses, std_losses


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price
    np.random.seed(42)
    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train)
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_losses, std_losses = fitting(X_train, y_train, X_test, y_test)
    percentages = np.arange(10, 101, 1)
    mean_loss = np.array(mean_losses)
    std_loss = np.array(std_losses)

    upper_bound = mean_loss + 2 * std_loss
    lower_bound = mean_loss - 2 * std_loss


    plt.figure()
    plt.fill_between(percentages, upper_bound, lower_bound, color='gray', alpha=0.3)
    plt.plot(percentages, mean_loss, color='blue', label='Average Loss')

    #plt.plot(percentages, mean_losses, label='Mean Loss')
    #plt.fill_between(percentages, lower_bound, upper_bound, alpha=0.2, color='gray',
    #                 label='Mean ± 2 STD')

    # plt.errorbar(np.arange(10, 101, 1), mean_losses, yerr=2 * np.array(std_losses))
    plt.xlabel("Percentage of training data")
    plt.ylabel("Mean loss over test set")
    plt.title("Mean loss over test set as function of training size")
    plt.show()
