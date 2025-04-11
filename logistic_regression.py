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

    def plot_regression(self, title="Logistic Regression Analysis", feature_names=None):
        """Create visualization of the logistic regression results.

        Args:
            title: Title for the plot
            feature_names: List of feature names for x-axis labels

        Returns:
            matplotlib figure object
        """
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(self.results['coefficients']))]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, y=1.02)

        # Plot feature coefficients
        axes = [ax1, ax2, ax3, ax4]
        for i, (name, coef) in enumerate(zip(feature_names, self.results['coefficients'])):
            x_range = np.linspace(-3, 3, 100)
            y = 1 / (1 + np.exp(-(coef * x_range + self.results['intercept'])))
            
            axes[i].plot(x_range, y, 'r-', label=f'Coefficient: {coef:.4f}')
            axes[i].set_title(f'{name} Impact')
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
