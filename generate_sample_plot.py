"""Generate sample plot for README documentation."""
import pandas as pd
from analyze_weather import analyze_weather

# Create sample weather data
data = {
    'Humidity': [0.65, 0.71, 0.80, 0.75, 0.68, 0.72, 0.79, 0.83],
    'Pressure (millibars)': [1010.2, 1008.5, 1012.3, 1009.8, 1011.0, 1007.5, 1013.2, 1010.8],
    'Temperature (C)': [20.5, 18.2, 15.8, 22.1, 19.4, 17.6, 16.2, 21.3],
    'Wind Speed (km/h)': [12.5, 15.8, 8.2, 10.1, 14.3, 11.7, 9.4, 13.6],
    'Precip Type': ['none', 'rain', 'rain', 'none', 'snow', 'rain', 'snow', 'none']
}

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('sample_weather.csv', index=False)

# Run analysis to generate plot
analyze_weather('sample_weather.csv')
