"""Test module for weather data analysis."""
import os
import pandas as pd
import pytest
import numpy as np
from analyze_weather import prepare_data, analyze_weather

@pytest.fixture(name='sample_weather_data')
def create_sample_weather_data():
    """Create sample weather data for testing.

    Returns:
        pd.DataFrame: Sample weather data
    """
    data = {
        'Humidity': [0.65, 0.71, 0.80, 0.75, 0.68],
        'Pressure (millibars)': [1010.2, 1008.5, 1012.3, 1009.8, 1011.0],
        'Temperature (C)': [20.5, 18.2, 15.8, 22.1, 19.4],
        'Wind Speed (km/h)': [12.5, 15.8, 8.2, 10.1, 14.3],
        'Precip Type': ['none', 'rain', 'rain', 'none', 'snow']
    }
    return pd.DataFrame(data)

def test_prepare_data(sample_weather_data):
    """Test if prepare_data function correctly processes weather data.

    Args:
        sample_weather_data: Fixture providing sample weather DataFrame
    """
    features, rain_target, snow_target = prepare_data(sample_weather_data)

    # Check features
    assert isinstance(features, pd.DataFrame)
    assert features.shape == (5, 4)
    assert not features.isna().any().any()

    # Check rain target
    assert isinstance(rain_target, pd.Series)
    assert len(rain_target) == 5
    assert set(rain_target.unique()) <= {0, 1}  # Binary values only
    assert not rain_target.isna().any()
    assert sum(rain_target) == 2  # Two rain instances in sample data

    # Check snow target
    assert isinstance(snow_target, pd.Series)
    assert len(snow_target) == 5
    assert set(snow_target.unique()) <= {0, 1}  # Binary values only
    assert not snow_target.isna().any()
    assert sum(snow_target) == 1  # One snow instance in sample data

    # Verify no overlap between rain and snow
    assert not ((rain_target == 1) & (snow_target == 1)).any()

    # Verify correct classification
    assert rain_target[sample_weather_data['Precip Type'] == 'rain'].all()
    assert snow_target[sample_weather_data['Precip Type'] == 'snow'].all()
    assert (rain_target[sample_weather_data['Precip Type'] == 'none'] == 0).all()
    assert (snow_target[sample_weather_data['Precip Type'] == 'none'] == 0).all()

def test_prepare_data_with_missing_values():
    """Test if prepare_data handles missing values correctly."""
    data = {
        'Humidity': [0.65, None, 0.80, 0.75, 0.68],
        'Pressure (millibars)': [1010.2, 1008.5, None, 1009.8, 1011.0],
        'Temperature (C)': [20.5, 18.2, 15.8, None, 19.4],
        'Wind Speed (km/h)': [None, 15.8, 8.2, 10.1, 14.3],
        'Precip Type': ['none', 'rain', None, 'none', 'snow']
    }
    df = pd.DataFrame(data)
    features, rain_target, snow_target = prepare_data(df)

    # Check that missing values are handled
    assert not features.isna().any().any()
    assert not rain_target.isna().any()
    assert not snow_target.isna().any()

    # Check binary values
    assert set(rain_target.unique()) <= {0, 1}
    assert set(snow_target.unique()) <= {0, 1}

    # Check that missing precip type is treated as 'none'
    assert rain_target[2] == 0  # Index where Precip Type is None
    assert snow_target[2] == 0  # Index where Precip Type is None

def test_analyze_weather_file_not_found(tmp_path):
    """Test analyze_weather handles missing file correctly."""
    # Create a temporary CSV file
    df = pd.DataFrame({
        'Humidity': [0.65],
        'Pressure (millibars)': [1010.2],
        'Temperature (C)': [20.5],
        'Wind Speed (km/h)': [12.5],
        'Precip Type': ['none']
    })
    csv_path = os.path.join(tmp_path, 'test_weather.csv')
    df.to_csv(csv_path, index=False)

    # Test with existing file
    analyze_weather()  # Should run without errors

    # Test with non-existent file
    os.remove(csv_path)
    analyze_weather()  # Should handle the error gracefully
