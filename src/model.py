import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import shap
import joblib
import matplotlib.pyplot as plt
import os


def load_data():
    df = pd.read_csv('data/featured_laps.csv')
    print(f"Loaded featured data: {df.shape}")
    print(f"Races: {df['Race'].unique()}")
    print(f"Years: {df['Year'].unique()}")
    return df


def prepare_data(df):

    features = [
    'LapNumber', 'Position', 'TyreLife', 'CompoundEncoded',
    'DegradationRate', 'PositionMomentum', 'RaceCompletion',
    'PitThisLap', 'PitStopDuration', 'StintNumber',
    'GapToLeaderPace', 'RelativeTyreAge', 'LapsSincePit',
    'GridPosition'
]

    target = 'FinalPosition'

    # train on 2021-2024, test on 2025
    train = df[df['Year'].isin([2021, 2022, 2023, 2024])].copy()
    test  = df[df['Year'] == 2025].copy()

    print(f"\nTraining years: 2021, 2022, 2023, 2024")
    print(f"Testing year:   2025")
    print(f"\nTraining rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")
    print(f"\nFeatures used: {features}")

    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    return X_train, y_train, X_test, y_test, features


def baseline_model(X_test, y_test):
    naive_predictions = X_test['Position'].values
    mae = mean_absolute_error(y_test, naive_predictions)
    print(f"\nBaseline MAE (predict current position): {mae:.2f} positions")
    return mae


def train_model(X_train, y_train, X_test, y_test):

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nLightGBM MAE: {mae:.2f} positions")

    predictions_rounded = np.round(predictions).astype(int)
    predictions_rounded = np.clip(predictions_rounded, 1, 20)

    actual_top5    = (y_test <= 5).values
    predicted_top5 = (predictions_rounded <= 5)
    top5_accuracy  = (actual_top5 == predicted_top5).mean()

    print(f"Top 5 Finish Accuracy: {top5_accuracy:.1%}")

    return predictions, mae


def explain_model(model, X_test, features):

    print("\nGenerating SHAP explanation...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test,
                      feature_names=features,
                      show=False)
    plt.title("Feature Importance — SHAP Summary")
    plt.tight_layout()

    os.makedirs('models', exist_ok=True)
    plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
    print("SHAP plot saved to models/shap_summary.png")
    plt.show()


def plot_predictions(X_test, y_test, predictions):

    results = X_test.copy()
    results['ActualFinal']    = y_test.values
    results['PredictedFinal'] = np.round(predictions).astype(int)
    results['Error']          = abs(
        results['ActualFinal'] - results['PredictedFinal']
    )

    plt.figure(figsize=(10, 5))
    plt.hist(results['Error'], bins=20,
             color='steelblue', edgecolor='white')
    plt.title("Prediction Error Distribution")
    plt.xlabel("Absolute Position Error")
    plt.ylabel("Number of Laps")
    plt.axvline(
        results['Error'].mean(), color='red',
        linestyle='--',
        label=f"Mean Error: {results['Error'].mean():.2f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/error_distribution.png', dpi=150)
    print("Error plot saved to models/error_distribution.png")
    plt.show()

    return results


if __name__ == "__main__":

    df = load_data()

    X_train, y_train, X_test, y_test, features = prepare_data(df)

    baseline_mae = baseline_model(X_test, y_test)

    print("\nTraining LightGBM...")
    model = train_model(X_train, y_train, X_test, y_test)

    predictions, lgbm_mae = evaluate_model(model, X_test, y_test)

    improvement = baseline_mae - lgbm_mae
    print(f"\nImprovement over baseline: {improvement:.2f} positions")

    explain_model(model, X_test, features)

    results = plot_predictions(X_test, y_test, predictions)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_model.pkl')
    print("\nModel saved to models/lgbm_model.pkl")

    results.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")