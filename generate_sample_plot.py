"""Generate sample plots for binary classification of precipitation types."""
import pandas as pd
import numpy as np
from analyze_weather import analyze_weather

# Create more comprehensive sample weather data
np.random.seed(42)  # For reproducibility

# Generate 30 samples
n_samples = 30
data = {
    'Humidity': np.random.uniform(0.3, 0.9, n_samples),
    'Pressure (millibars)': np.random.uniform(980, 1020, n_samples),
    'Temperature (C)': np.random.uniform(-5, 25, n_samples),
    'Wind Speed (km/h)': np.random.uniform(0, 30, n_samples),
}

# Create precipitation types with realistic conditions
data['Precip Type'] = ['none'] * n_samples

# Assign rain (warm temperatures, high humidity)
rain_mask = (
    (data['Temperature (C)'] > 5) &
    (data['Humidity'] > 0.7)
)
data['Precip Type'] = np.where(rain_mask, 'rain', data['Precip Type'])

# Assign snow (cold temperatures, moderate humidity)
snow_mask = (
    (data['Temperature (C)'] < 2) &
    (data['Humidity'] > 0.6)
)
data['Precip Type'] = np.where(snow_mask, 'snow', data['Precip Type'])

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('sample_weather.csv', index=False)

# Print data distribution
print("\nData Distribution:")
print(df['Precip Type'].value_counts())

# Run analysis to generate plots
analyze_weather('sample_weather.csv')
