"""Logistic Regression Analysis for Weather Prediction."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

class LogisticRegressionAnalysis:
    """Logistic Regression model for binary classification of precipitation types."""

    def __init__(self):
        """Initialize the logistic regression model."""
        self.model = LogisticRegression(random_state=42)
        self.results = {}

    def fit(self, input_features, target_values):
        """Fit the logistic regression model and compute performance metrics.

        Args:
            input_features: DataFrame of feature variables
            target_values: Series of binary target values (0 or 1)
        """
        # Fit the model
        self.model.fit(input_features, target_values)

        # Make predictions
        y_pred = self.model.predict(input_features)
        y_pred_proba = self.model.predict_proba(input_features)[:, 1]

        # Calculate metrics
        self.results['accuracy'] = accuracy_score(target_values, y_pred)
        self.results['precision'] = precision_score(target_values, y_pred)
        self.results['recall'] = recall_score(target_values, y_pred)
        self.results['f1'] = f1_score(target_values, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(target_values, y_pred_proba)
        self.results['roc_curve'] = (fpr, tpr)
        self.results['auc'] = auc(fpr, tpr)

        # Store coefficients and intercept
        self.results['coefficients'] = self.model.coef_[0]
        self.results['intercept'] = self.model.intercept_[0]

    def predict_proba(self, input_features):
        """Predict probability of precipitation.

        Args:
            input_features: DataFrame of feature variables

        Returns:
            Array of predicted probabilities
        """
        return self.model.predict_proba(input_features)[:, 1]

    def plot_regression(self, title="Logistic Regression Analysis", feature_names=None, input_features=None, target_values=None):
        """Create visualization of the logistic regression results with probability curves.

        Args:
            title: Title for the plot
            feature_names: List of feature names for x-axis labels
            input_features: DataFrame of feature variables used for training
            target_values: Series of binary target values used for training

        Returns:
            matplotlib figure object
        """
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(self.results['coefficients']))]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, y=1.02)

        # Plot feature coefficients and probability curves
        axes = [ax1, ax2, ax3, ax4]
        for i, (name, coef) in enumerate(zip(feature_names, self.results['coefficients'])):
            if input_features is not None and target_values is not None:
                # Get the current feature values
                x_data = input_features.iloc[:, i]
                
                # Create a range for the probability curve
                x_min, x_max = x_data.min(), x_data.max()
                x_range = np.linspace(x_min - 0.1 * (x_max - x_min), 
                                    x_max + 0.1 * (x_max - x_min), 100)
                
                # Calculate other features' mean values
                other_features = input_features.drop(input_features.columns[i], axis=1)
                other_features_mean = other_features.mean()
                
                # Create input matrix for prediction
                X_pred = np.zeros((len(x_range), input_features.shape[1]))
                other_idx = 0
                for j in range(input_features.shape[1]):
                    if j == i:
                        X_pred[:, j] = x_range
                    else:
                        X_pred[:, j] = other_features_mean.iloc[other_idx]
                        other_idx += 1
                
                # Calculate probabilities
                y_pred = self.model.predict_proba(X_pred)[:, 1]
                
                # Plot probability curve
                axes[i].plot(x_range, y_pred, 'r-', 
                           label=f'Probability curve\nCoefficient: {coef:.4f}')
                
                # Plot actual data points
                axes[i].scatter(x_data[target_values == 0], np.zeros(sum(target_values == 0)),
                              c='blue', alpha=0.5, label='Negative class')
                axes[i].scatter(x_data[target_values == 1], np.ones(sum(target_values == 1)),
                              c='red', alpha=0.5, label='Positive class')
            
            axes[i].set_title(f'{name} Impact on Probability')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Probability')
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        return fig

    def plot_roc_curve(self):
        """Plot the ROC curve.

        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        fpr, tpr = self.results['roc_curve']
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {self.results["auc"]:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig
