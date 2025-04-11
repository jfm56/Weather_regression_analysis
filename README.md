# Weather Data Linear Regression Analysis

This project analyzes weather data to predict precipitation type (none/rain/snow) using multiple environmental factors. The analysis uses a comprehensive linear regression model that provides detailed statistical measures and visualizations.

## Problem Formulation

### Variables
- **Dependent Variable (DV)**:
  - Precipitation Type (categorical: none, rain, snow)
  - Encoded as numeric values using LabelEncoder
  - Note: While categorical, using linear regression helps understand relationships

- **Independent Variables (IV)**:
  1. Humidity (continuous, range: 0-1)
  2. Pressure (continuous, millibars)
  3. Temperature (continuous, Celsius)
  4. Wind Speed (continuous, km/h)

### Analysis Approach
- **Method**: Multiple Linear Regression
- **Purpose**: Understand relationships between weather conditions and precipitation
- **Note**: For actual prediction tasks, a classification model would be more appropriate

## Analysis Results

### Model Performance
- R² Score: 0.33 (33% of variance explained)
- F-statistic: 11,877.60
- p-value: 1.11e-16

### Sum of Squares
- Total (TSS): 10,151.40
- Regression (RSS): 3,350.25
- Error (ESS): 6,801.14

### Feature Coefficients
- Humidity: -0.3994 (strongest effect)
- Temperature: -0.0236 (moderate effect)
- Wind Speed: -0.0055 (small effect)
- Pressure: ~0.0000 (minimal effect)
- Intercept: 1.7440

### Visualization
![Regression Analysis](weather_regression.png)

The plots above show the relationship between each predictor variable and precipitation type. Each subplot demonstrates how the individual feature correlates with the precipitation type, with the red line showing the trend.

### Detailed Interpretation

#### 1. Model Fit Statistics
- **R² = 0.33 (33% of variance explained)**
  - Indicates moderate predictive power
  - Suggests other important factors may not be captured
  - Typical for complex weather phenomena

- **F-statistic = 11,877.60 (p-value < 0.001)**
  - Model is highly statistically significant
  - Relationships are not due to chance

#### 2. Feature Impacts

1. **Humidity (β = -0.3994)**
   - Strongest predictor
   - Negative relationship with precipitation type
   - For every 1 unit increase in humidity:
     - Precipitation type score decreases by 0.3994
   - Suggests higher humidity associates with rain/snow

2. **Temperature (β = -0.0236)**
   - Second strongest predictor
   - Negative relationship
   - Each 1°C increase:
     - Decreases precipitation type score by 0.0236
   - Aligns with weather patterns: lower temperatures more likely for snow

3. **Wind Speed (β = -0.0055)**
   - Weak negative relationship
   - Minor influence on precipitation type
   - Practical impact is minimal

4. **Pressure (β ≈ 0)**
   - Negligible impact
   - Suggests local pressure may not directly influence precipitation type

#### 3. Practical Applications

1. **Weather Conditions Most Likely to Predict:**
   - Snow: Low humidity, low temperature
   - Rain: Moderate humidity, moderate temperature
   - No Precipitation: High humidity, high temperature

2. **Model Limitations:**
   - Moderate R² suggests limited predictive power
   - Linear regression may not capture non-linear weather patterns
   - Categorical outcome might be better served by classification models

3. **Recommendations:**
   - Consider using classification models for better prediction
   - Include additional features (dew point, cloud cover, seasonal variables)
   - Explore non-linear relationships
   - Collect more detailed weather data

## Data Setup

### Required Data Files
- `weatherHistory.csv`: Historical weather data containing:
  - Temperature (C)
  - Humidity
  - Pressure (millibars)
  - Wind Speed (km/h)
  - Precipitation Type (none/rain/snow)

### Data Installation
1. Download the weather history dataset
2. Place `weatherHistory.csv` in your project directory or update the file path in `analyze_weather.py`:
```python
# In analyze_weather.py, update this line with your data path:
df = pd.read_csv('path/to/your/weatherHistory.csv')
```

### Data Format Requirements
The CSV file should contain the following columns:
- `Temperature (C)`: Temperature in Celsius
- `Humidity`: Relative humidity (0-1)
- `Pressure (millibars)`: Atmospheric pressure
- `Wind Speed (km/h)`: Wind speed in kilometers per hour
- `Precip Type`: Precipitation type (none/rain/snow)

### Data Preprocessing
1. Missing Values:
   - Numeric columns (Temperature, Humidity, etc.) are filled with column means
   - Categorical columns (Precip Type) are filled with 'none'

2. Data Cleaning:
   - The script automatically handles:
     - Missing value imputation
     - Feature scaling (if needed)
     - Categorical encoding for precipitation types

3. Data Validation:
   - Ensures all required columns are present
   - Checks data types and converts if necessary
   - Validates value ranges (e.g., Humidity between 0-1)

### Troubleshooting Data Issues
If you encounter data-related errors:
1. Check column names match exactly (case-sensitive)
2. Ensure numeric columns contain valid numbers
3. Verify precipitation types are correctly formatted
4. Look for and remove any corrupt rows

Example of checking data quality:
```python
# Load and inspect data
import pandas as pd

df = pd.read_csv('weatherHistory.csv')

# Check for missing values
print(df.isnull().sum())

# Check unique values in Precip Type
print(df['Precip Type'].unique())

# Verify data ranges
print(df.describe())
```

## Project Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Analysis

### Basic Usage
```python
# Make sure weatherHistory.csv is in the correct location
from analyze_weather import analyze_weather

# Run the analysis
analyze_weather()
```

### Custom Analysis
```python
from analyze_weather import analyze_weather

# Run the analysis
# Create instance of the analysis class
model = LinearRegressionAnalysis()

# Load your own data
df = pd.read_csv('your_weather_data.csv')

# Prepare features (X) and target (y)
X = df[['Humidity', 'Pressure (millibars)', 'Temperature (C)', 'Wind Speed (km/h)']]
y = df['Precip Type']

# Fit the model and get results
model.fit(X, y)

# Print results
print(f"R² Score: {model.results['r2']:.3f}")
print(f"F-statistic: {model.results['f_statistic']:.3f}")

# Create and save visualization
plt = model.plot_regression("Your Analysis Title")
plt.savefig('your_analysis.png')
```

## Testing

Run tests with coverage:
```bash
pytest --cov=. --cov-report=term-missing
```

Run pylint:
```bash
pylint *.py
```

## CI/CD Pipeline

### GitHub Actions Setup

1. **Required Secrets**
   Navigate to your repository's Settings > Secrets and variables > Actions and add:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_TOKEN`: Your Docker Hub access token (not your password)

2. **Docker Hub Token**
   To get a Docker Hub access token:
   1. Log in to [Docker Hub](https://hub.docker.com)
   2. Go to Account Settings > Security
   3. Click "New Access Token"
   4. Give it a description (e.g., "GitHub Actions")
   5. Copy the token and save it as `DOCKER_TOKEN` in GitHub secrets

### Pipeline Features

1. **Automated Testing**
   - Runs pytest with coverage reporting
   - Uploads coverage reports to Codecov
   - Ensures test coverage stays high

2. **Code Quality**
   - Runs pylint
   - Enforces a minimum score of 9.0/10
   - Checks for code style and potential issues

3. **Docker Integration**
   - Builds Docker image on successful tests
   - Pushes to Docker Hub with tags:
     - `latest`: Most recent main branch build
     - `sha-xxxxx`: Git commit specific build
   - Uses layer caching for faster builds

## Local Development

### Docker Support

Build and run locally:
```bash
docker build -t weather-analysis .
docker run weather-analysis
```

Pull from Docker Hub:
```bash
docker pull $DOCKER_USERNAME/weather-analysis:latest
docker run $DOCKER_USERNAME/weather-analysis:latest
```
