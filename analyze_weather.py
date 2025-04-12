"""Weather Data Analysis Script for predicting precipitation types using multiple features."""
import pandas as pd
from logistic_regression import LogisticRegressionAnalysis

def prepare_data(dataframe):
    """Prepare weather data for binary classification analysis.

    Args:
        dataframe: Pandas DataFrame containing weather data

    Returns:
        tuple: (features, rain_target, snow_target)
    """
    # Fill missing values in precipitation type
    dataframe['Precip Type'] = dataframe['Precip Type'].fillna('none')

    # Create binary targets for rain and snow
    rain_target = (dataframe['Precip Type'] == 'rain').astype(int)
    snow_target = (dataframe['Precip Type'] == 'snow').astype(int)

    # Prepare features
    features = dataframe[[
        'Humidity',
        'Pressure (millibars)',
        'Temperature (C)',
        'Wind Speed (km/h)'
    ]].copy()

    # Handle any missing values in features
    features = features.fillna(features.mean())

    return features, rain_target, snow_target

def analyze_weather(file_path='/Users/jimmullen/Downloads/weatherHistory.csv'):
    """Perform binary classification analysis for rain and snow prediction.

    Args:
        file_path: Path to the weather data CSV file
    """
    try:
        # Load and prepare data
        dataframe = pd.read_csv(file_path)
        features, rain_target, snow_target = prepare_data(dataframe)

        # Create and fit models for rain and snow
        rain_model = LogisticRegressionAnalysis()
        snow_model = LogisticRegressionAnalysis()

        rain_model.fit(features, rain_target)
        snow_model.fit(features, snow_target)

        # Print results for rain prediction
        print("\nRain Prediction Model:")
        print("-" * 50)
        print("\nModel Performance:")
        print(f"Accuracy: {rain_model.results['accuracy']:.4f}")
        print(f"Precision: {rain_model.results['precision']:.4f}")
        print(f"Recall: {rain_model.results['recall']:.4f}")
        print(f"F1 Score: {rain_model.results['f1']:.4f}")
        print(f"AUC-ROC: {rain_model.results['auc']:.4f}")

        print("\nFeature Coefficients (Rain):")
        feature_names = ['Humidity', 'Pressure', 'Temperature', 'Wind Speed']
        for name, coef in zip(feature_names, rain_model.results['coefficients']):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {rain_model.results['intercept']:.4f}")

        # Print results for snow prediction
        print("\nSnow Prediction Model:")
        print("-" * 50)
        print("\nModel Performance:")
        print(f"Accuracy: {snow_model.results['accuracy']:.4f}")
        print(f"Precision: {snow_model.results['precision']:.4f}")
        print(f"Recall: {snow_model.results['recall']:.4f}")
        print(f"F1 Score: {snow_model.results['f1']:.4f}")
        print(f"AUC-ROC: {snow_model.results['auc']:.4f}")

        print("\nFeature Coefficients (Snow):")
        for name, coef in zip(feature_names, snow_model.results['coefficients']):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {snow_model.results['intercept']:.4f}")

        # Create visualizations
        plot_names = ['Humidity', 'Pressure (mb)', 'Temperature (C)', 'Wind Speed (km/h)']

        # Feature impact plots with probability curves
        rain_plot = rain_model.plot_regression(
            "Rain Prediction Model", plot_names,
            input_features=features, target_values=rain_target)
        rain_plot.savefig('images/rain_feature_impact.png',
                         bbox_inches='tight', dpi=300)

        snow_plot = snow_model.plot_regression(
            "Snow Prediction Model", plot_names,
            input_features=features, target_values=snow_target)
        snow_plot.savefig('images/snow_feature_impact.png',
                         bbox_inches='tight', dpi=300)

        # ROC curves
        rain_roc = rain_model.plot_roc_curve()
        rain_roc.savefig('images/rain_roc_curve.png',
                        bbox_inches='tight', dpi=300)

        snow_roc = snow_model.plot_roc_curve()
        snow_roc.savefig('images/snow_roc_curve.png',
                        bbox_inches='tight', dpi=300)

        print("\nVisualizations saved:")
        print("- Feature impact: 'rain_feature_impact.png' and 'snow_feature_impact.png'")
        print("- ROC curves: 'rain_roc_curve.png' and 'snow_roc_curve.png'")

    except FileNotFoundError as file_err:
        print(f"Error: Could not find weather data file - {str(file_err)}")
    except pd.errors.EmptyDataError as data_err:
        print(f"Error: Data file is empty - {str(data_err)}")
    except (AttributeError, KeyError) as struct_err:
        print(f"Error: Data structure is invalid - {str(struct_err)}")
    except ValueError as val_err:
        print(f"Error: Invalid data format - {str(val_err)}")

if __name__ == "__main__":
    analyze_weather()
