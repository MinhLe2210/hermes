import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    daily = (
        df.groupby('date')['delay_minutes']
        .mean()
        .reset_index()
        .rename(columns={'delay_minutes': 'delay_rate'})
    )
    daily['dayofweek'] = daily['date'].dt.dayofweek
    daily['month'] = daily['date'].dt.month
    daily['rolling_mean_7'] = daily['delay_rate'].rolling(window=7, min_periods=1).mean()
    daily['rolling_mean_14'] = daily['delay_rate'].rolling(window=14, min_periods=1).mean()
    daily['target'] = daily['delay_rate']
    return daily


def train_linear_delay_model(daily: pd.DataFrame):
    X = daily[['dayofweek', 'month', 'rolling_mean_7', 'rolling_mean_14']]
    y = daily['target']
    model = LinearRegression()
    model.fit(X, y)

    last_row = daily.iloc[-1]
    next_days = pd.date_range(start=last_row['date'] + pd.Timedelta(days=1), periods=7)
    next_df = pd.DataFrame({'date': next_days})
    next_df['dayofweek'] = next_df['date'].dt.dayofweek
    next_df['month'] = next_df['date'].dt.month
    next_df['rolling_mean_7'] = last_row['rolling_mean_7']
    next_df['rolling_mean_14'] = last_row['rolling_mean_14']

    y_pred_next = model.predict(next_df[['dayofweek', 'month', 'rolling_mean_7', 'rolling_mean_14']])
    results = pd.DataFrame({'date': next_df['date'], 'predicted_delay_rate': y_pred_next})
    return model, results


def evaluate_model(daily: pd.DataFrame):
    train = daily.iloc[:-7]
    test = daily.iloc[-7:]
    X_train = train[['dayofweek', 'month', 'rolling_mean_7', 'rolling_mean_14']]
    y_train = train['target']
    X_test = test[['dayofweek', 'month', 'rolling_mean_7', 'rolling_mean_14']]
    y_test = test['target']
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "R2": r2}


def predict_model():
    csv_path = os.getcwd() + "/data/shipment.csv"
    daily = prepare_data(csv_path)
    metrics = evaluate_model(daily)
    model, results = train_linear_delay_model(daily)
    print("Model metrics:", metrics)
    print("\nNext week delay predictions:")
    print(results)
    return results


if __name__ == "__main__":
    predict_model()

# csv_path = "/home/minh/project/ship_agent/data/shipment.csv"