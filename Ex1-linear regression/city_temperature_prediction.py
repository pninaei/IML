
import pandas as pd
import plotly.express as px
import polynomial_fitting
from sklearn.model_selection import train_test_split


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load the dataset, parsing the 'Date' column as datetime
    data = pd.read_csv(filename, parse_dates=['Date'])

    # Drop rows with invalid data (e.g., NaN values)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data = data[data.Temp > -20]

    # Add 'DayOfYear' column based on the 'Date' column
    data['DayOfYear'] = data['Date'].dt.dayofyear

    return data


def israel_temp_relation(df):

    data = df[df["Country"] == "Israel"]
    temperature = data['Temp']
    day_of_year = data['DayOfYear']
    px.scatter(data, x=day_of_year, y=temperature, color="Year",
               title='the relation between the day of the year and the temperature in Israel').show()

    monthly_std = data.groupby('Month').agg({'Temp': 'std'}).reset_index()
    px.bar(monthly_std, title="The std of daily temperature in each month over the years", x="Month",
           y="Temp").show()


def avg_monthly_temp_with_error_bars(df):
    # Group by 'Country' and 'Month', then calculate the mean and standard deviation of 'Temp'
    monthly_stats = df.groupby(['Country', 'Month']).agg(
        avg_temp=('Temp', 'mean'),
        std_temp=('Temp', 'std')
    ).reset_index()

    # Create the line plot with error bars
    fig = px.line(
        monthly_stats,
        x='Month',
        y='avg_temp',
        color='Country',
        error_y='std_temp',
        title='Average Monthly Temperature with Error Bars by Country')

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Average Temperature'
    )
    fig.show()


def fit_polynomial_models(df):

    df = df[df['Country'] == "Israel"]
    X, y = df["DayOfYear"], df["Temp"]

    # Split X and y into training and test sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    test_errors = []

    for k in range(1, 11):
        model = polynomial_fitting.PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        rounded = round(loss, 2)
        print(f"Degree {k} has {round(loss, 2)} error")
        test_errors.append(rounded)

    degrees = list(range(1, 11))

    fig = px.bar(x=degrees, y=test_errors, labels={'x': 'Polynomial Degree (k)', 'y': 'Test Error'},
                 title='Test Error vs. Polynomial Degree')
    fig.update_layout(xaxis=dict(tickmode='linear'))
    fig.show()


def evaluate_model_on_countries(df, best_k):
    # Fit model using the best degree on the entire Israel data
    data = df[df['Country'] == "Israel"]
    X_train, y_train = data["DayOfYear"], data["Temp"]

    model = polynomial_fitting.PolynomialFitting(best_k)
    model.fit(X_train, y_train)

    # Evaluate the model on each country
    countries = df['Country'].unique()
    errors = {}

    for country in countries:
        if country != "Israel":
            country_data = df[df['Country'] == country]
            X_country, y_country = country_data["DayOfYear"], country_data["Temp"]
            loss = model.loss(X_country, y_country)
            errors[country] = loss

    # Plot the errors for each country
    countries = list(errors.keys())
    losses = list(errors.values())
    fig = px.bar(x=countries, y=losses, labels={'x': 'Country', 'y': 'Model Error'},
                 title=f'Model Error on Different Countries (Degree {best_k})')
    fig.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset

    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_temp_relation(df)
    # Question 4 - Exploring differences between countries
    avg_monthly_temp_with_error_bars(df)
    # Question 5 - Fitting model for different values of `k`
    fit_polynomial_models(df)
    # Question 6 - Evaluating fitted model on different countries
    best_k = 5  # Choose the best k based on previous results
    evaluate_model_on_countries(df, best_k)


