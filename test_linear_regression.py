"""
Test module for linear regression analysis.
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionAnalysis

@pytest.fixture(name='regression_data')
def create_sample_data():
    """Create sample data for testing regression analysis.

    Returns:
        tuple: (features, target) where features is a 2D array and target is a 1D array
    """
    np.random.seed(42)
    features = np.linspace(0, 10, 100).reshape(-1, 1)
    target = 2 * features.ravel() + 1 + np.random.normal(0, 1, 100)
    return features, target

def test_model_fitting(regression_data):
    """Test if the model fits correctly and computes all required statistics.

    Args:
        regression_data: Fixture providing sample features and target data
    """
    features, target = regression_data
    model = LinearRegressionAnalysis()
    model.fit(features, target)

    assert isinstance(model.results['total_sum_squares'], float)
    assert isinstance(model.results['regression_sum_squares'], float)
    assert isinstance(model.results['error_sum_squares'], float)
    assert isinstance(model.results['r2'], float)
    assert isinstance(model.results['f_statistic'], float)
    assert isinstance(model.results['p_value'], float)
    assert isinstance(model.results['intercept'], float)
    assert isinstance(model.results['coefficients'], np.ndarray)

def test_r2_score_range(regression_data):
    """Test if R2 score is within valid range [0, 1].

    Args:
        regression_data: Fixture providing sample features and target data
    """
    features, target = regression_data
    model = LinearRegressionAnalysis()
    model.fit(features, target)

    assert 0 <= model.results['r2'] <= 1

def test_sums_of_squares_relation(regression_data):
    """Test if sums of squares follow the expected relation.

    Total Sum of Squares should equal Regression Sum of Squares plus Error Sum of Squares.

    Args:
        regression_data: Fixture providing sample features and target data
    """
    features, target = regression_data
    model = LinearRegressionAnalysis()
    model.fit(features, target)

    np.testing.assert_almost_equal(
        model.results['total_sum_squares'],
        model.results['regression_sum_squares'] + model.results['error_sum_squares'],
        decimal=10
    )

def test_plot_regression(regression_data):
    """Test if plot_regression method works correctly.

    Args:
        regression_data: Fixture providing sample features and target data
    """
    features, target = regression_data
    model = LinearRegressionAnalysis()
    model.fit(features, target)

    # Test with default parameters
    plt_obj = model.plot_regression()
    assert plt_obj is not None
    plt.close()

    # Test with custom title and feature names
    plt_obj = model.plot_regression(title="Test Plot", feature_names=["Feature 1"])
    assert plt_obj is not None
    plt.close()

    # Test error case when model is not fitted
    model_unfitted = LinearRegressionAnalysis()
    with pytest.raises(ValueError, match="Model must be fitted before plotting"):
        model_unfitted.plot_regression()
