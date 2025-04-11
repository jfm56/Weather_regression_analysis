"""Weather Data Analysis Script for predicting precipitation types using multiple features."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from linear_regression import LinearRegressionAnalysis

def prepare_data(dataframe):
    """Prepare weather data for regression analysis.

    Args:
        dataframe: Pandas DataFrame containing weather data

    Returns:
        tuple: (features, target, precipitation_classes)
    """
    # Convert Precip Type to numeric using label encoding
    label_encoder = LabelEncoder()
    dataframe['Precip Type Numeric'] = label_encoder.fit_transform(
        dataframe['Precip Type'].fillna('none')
    )

    # Prepare features
    features = dataframe[[
        'Humidity',
        'Pressure (millibars)',
        'Temperature (C)',
        'Wind Speed (km/h)'
    ]].copy()

    # Handle any missing values
    features = features.fillna(features.mean())
    target = dataframe['Precip Type Numeric']

    return features, target, label_encoder.classes_

def analyze_weather(file_path='/Users/jimmullen/Downloads/weatherHistory.csv'):
    """Perform regression analysis on weather data and generate visualizations.

    Args:
        file_path: Path to the weather data CSV file
    """
    try:
        # Load and prepare data
        dataframe = pd.read_csv(file_path)
        features, target, precip_classes = prepare_data(dataframe)

        # Create and fit the model
        model = LinearRegressionAnalysis()
        model.fit(features, target)

        # Print results
        print("\nRegression Analysis Results:")
        print("-" * 50)
        print(f"Precipitation Types: {precip_classes}")
        print("\nModel Statistics:")
        print(f"R² Score: {model.results['r2']:.4f}")
        print(f"F-statistic: {model.results['f_statistic']:.4f}")
        print(f"p-value: {model.results['p_value']:.4e}")
        print("\nSum of Squares:")
        print(f"Total (TSS): {model.results['total_sum_squares']:.4f}")
        print(f"Regression (RSS): {model.results['regression_sum_squares']:.4f}")
        print(f"Error (ESS): {model.results['error_sum_squares']:.4f}")

        print("\nModel Coefficients:")
        feature_names = ['Humidity', 'Pressure', 'Temperature', 'Wind Speed']
        for name, coef in zip(feature_names, model.results['coefficients']):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {model.results['intercept']:.4f}")

        # Create visualization
        plot_names = ['Humidity', 'Pressure (mb)', 'Temperature (C)', 'Wind Speed (km/h)']
        regression_plot = model.plot_regression(
            "Precipitation Type Prediction", plot_names)
        regression_plot.savefig('weather_regression.png',
                              bbox_inches='tight', dpi=300)
        print("\nVisualization saved as 'weather_regression.png'")

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
