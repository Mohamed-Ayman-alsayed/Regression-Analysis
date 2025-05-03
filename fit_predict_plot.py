import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None  # Changed from 'weights' to 'coefficients'
        self.residuals = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit model using Normal Equation"""
        # Add intercept term
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Calculate coefficients using normal equation
        X_transpose = X_with_intercept.T
        XtX = np.dot(X_transpose, X_with_intercept)
        XtY = np.dot(X_transpose, y)
        self.coefficients = np.linalg.inv(XtX).dot(XtY)
        
        # Store residuals and training data
        y_pred = np.dot(X_with_intercept, self.coefficients)
        self.residuals = y - y_pred
        self.X_train = X_with_intercept
        self.y_train = y

    def predict(self, X):
        """Make predictions on new data"""
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")
        
        # Add intercept if needed
        if X.shape[1] == len(self.coefficients) - 1:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            
        return np.dot(X, self.coefficients)

    def sum_of_squared_errors(self, y, pred):
        """Calculate SSE between actual and predicted values"""
        return np.sum((y - pred) ** 2)
    
    

    def anova(self):
        """Calculate ANOVA table for regression"""
        if self.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        n = self.X_train.shape[0]
        k = self.X_train.shape[1] - 1  # Exclude intercept
        
        y_pred = self.predict(self.X_train[:, 1:])  # Exclude intercept column
        y_mean = np.mean(self.y_train)
        
        # Calculate sums of squares
        SST = np.sum((self.y_train - y_mean) ** 2)
        SSR = np.sum((y_pred - y_mean) ** 2)
        SSE = self.sum_of_squared_errors(self.y_train, y_pred)
        
        # Degrees of freedom
        df_reg = k
        df_resid = n - k - 1
        df_total = n - 1
        
        # Mean squares
        MSR = SSR / df_reg
        MSE = SSE / df_resid
        
        # F-statistic and p-value
        F = MSR / MSE
        p_value = stats.f.sf(F, df_reg, df_resid)
        
        # Create ANOVA table
        return pd.DataFrame({
            'Source': ['Regression', 'Residual', 'Total'],
            'SS': [SSR, SSE, SST],
            'df': [df_reg, df_resid, df_total],
            'MS': [MSR, MSE, np.nan],
            'F': [F, np.nan, np.nan],
            'p-value': [p_value, np.nan, np.nan]
        })
        
        
    """Hossam's code"""    
        
    def hypothesis_test(self):
        """Perform hypothesis testing for each coefficient (t-test)"""
        if self.coefficients is None:
            raise ValueError("Model must be fitted first")
    
        n, k_plus_1 = self.X_train.shape  # k+1 includes intercept
        k = k_plus_1 - 1  # exclude intercept for degree of freedom
        df_resid = n - k_plus_1

        # Estimate variance of residuals
        y_pred = self.predict(self.X_train[:, 1:])  # exclude intercept in prediction
        residuals = self.y_train - y_pred
        s_squared = np.sum(residuals ** 2) / df_resid

        # Variance-covariance matrix of coefficients
        XtX_inv = np.linalg.inv(np.dot(self.X_train.T, self.X_train))
        var_betas = s_squared * np.diag(XtX_inv)

        # Standard errors
        se_betas = np.sqrt(var_betas)

        # t-statistics
        t_stats = self.coefficients / se_betas

        # two-tailed p-values
        from scipy.stats import t
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=df_resid))

        # Build a summary table
        summary = pd.DataFrame({
            'Coefficient': self.coefficients,
            'Standard Error': se_betas,
            't-statistic': t_stats,
            'p-value': p_values
        })
    
        return summary
    



    def plot(self, X, y):
        """3D plot for models with exactly 2 features"""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] != 2:  # Fixed from original 12 to 2
            raise ValueError("Plotting requires exactly two features")
        
        # Create grid for surface plot
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        # Calculate predicted values for grid
        grid_matrix = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        y_grid = self.predict(grid_matrix).reshape(x1_grid.shape)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface and data points
        ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='yellow')
        ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual Data')
        
        # Add labels and legend
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target")
        ax.set_title("Regression Plane vs Actual Data Points")
        plt.legend()
        plt.show()
        
        
print("Plotting complete")