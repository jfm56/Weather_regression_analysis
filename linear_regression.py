"""
Linear Regression Analysis Module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

class LinearRegressionAnalysis:
    """Class for performing comprehensive linear regression analysis."""

    def __init__(self):
        """Initialize regression analysis with empty model and results."""
        self.model = LinearRegression()
        self.input_features = None
        self.target_values = None
        self.predictions = None
        self.results = {}

    def fit(self, features, target):
        """
        Fit the linear regression model and compute all statistics.

        Args:
            features: Independent variable(s)
            target: Dependent variable
        """
        if isinstance(features, pd.Series):
            features = features.values.reshape(-1, 1)
        if isinstance(target, pd.Series):
            target = target.values

        self.input_features = features
        self.target_values = target

        # Fit the model
        self.model.fit(features, target)
        self.predictions = self.model.predict(features)

        # Calculate sums of squares
        self.results['total_sum_squares'] = np.sum((target - np.mean(target))**2)
        self.results['regression_sum_squares'] = np.sum((self.predictions - np.mean(target))**2)
        self.results['error_sum_squares'] = np.sum((target - self.predictions)**2)

        # Calculate R-squared
        self.results['r2'] = r2_score(target, self.predictions)

        # Calculate F-test and p-value
        n_samples = len(target)
        n_features = features.shape[1]
        f_stat = (self.results['regression_sum_squares']/n_features) / \
                 (self.results['error_sum_squares']/(n_samples-n_features-1))
        self.results['f_statistic'] = f_stat
        self.results['p_value'] = 1 - stats.f.cdf(f_stat, n_features, n_samples-n_features-1)

        # Store coefficients
        self.results['intercept'] = self.model.intercept_
        self.results['coefficients'] = self.model.coef_

        return self

    def plot_regression(self, title="Linear Regression Analysis", feature_names=None):
        """
        Create visualization for regression analysis.

        Args:
            title: Plot title
            feature_names: List of feature names for independent variables

        Returns:
            matplotlib.pyplot: Plot object with regression visualizations
        """
        if self.input_features is None or self.target_values is None:
            raise ValueError("Model must be fitted before plotting")

        # Create subplots for each feature
        n_features = self.input_features.shape[1]
        _, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
        if n_features == 1:
            axes = [axes]

        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]

        for i, (ax, name) in enumerate(zip(axes, feature_names)):
            feature_values = (self.input_features.iloc[:, i]
                            if isinstance(self.input_features, pd.DataFrame)
                            else self.input_features[:, i])
            # Plot scatter points
            ax.scatter(feature_values, self.target_values,
                      color='blue', alpha=0.5, s=20)
            # Calculate and plot trend line
            coef = self.results['coefficients'][i]
            intercept = self.results['intercept']
            x_range = np.linspace(np.min(feature_values),
                                np.max(feature_values), 100)
            y_pred = coef * x_range + intercept
            ax.plot(x_range, y_pred, color='red',
                    label=f'Coef: {coef:.3f}')
            ax.set_xlabel(name)
            if i == 0:
                ax.set_ylabel('Target Variable')
            ax.legend()

        plt.suptitle(f'{title}\nR² = {self.results["r2"]:.3f}')
        plt.tight_layout()
        return plt
