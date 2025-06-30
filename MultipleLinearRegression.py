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
    
    

    def anova(self, X, y, feature_names=None):
        """Calculate ANOVA table for simple regression on each feature separately"""
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]

        for idx in range(X.shape[1]):
            Xi = X[:, idx].reshape(-1, 1)  # take one feature
            self.fit(Xi, y)

            n = Xi.shape[0]
            y_pred = self.predict(Xi)
            y_mean = np.mean(y)

            # Calculate sums of squares
            SST = np.sum((y - y_mean) ** 2)
            SSR = np.sum((y_pred - y_mean) ** 2)
            SSE = np.sum((y - y_pred) ** 2)

            # Degrees of freedom
            df_reg = 1
            df_resid = n - 2
            df_total = n - 1

            # Mean squares
            MSR = SSR / df_reg
            MSE = SSE / df_resid

            # F-statistic and p-value
            F = MSR / MSE
            p_value = stats.f.sf(F, df_reg, df_resid)

            # Output
            print(f"ANOVA for Feature: {feature_names[idx]}")
            print(pd.DataFrame({
                'Source': ['Regression', 'Residual', 'Total'],
                'SS': [SSR, SSE, SST],
                'df': [df_reg, df_resid, df_total],
                'MS': [MSR, MSE, None],
                'F': [F, None, None],
                'p-value': [p_value, None, None]
            }))

        
        
       
    
    def hypothesis_test(self, X, y, feature_names=None):
        """Perform hypothesis testing for simple regression on each feature separately"""
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]

        for idx in range(X.shape[1]):
            Xi = X[:, idx].reshape(-1, 1)  # take one feature
            self.fit(Xi, y)

            n = Xi.shape[0]
            df_resid = n - 2

            y_pred = self.predict(Xi)
            residuals = y - y_pred
            s_squared = np.sum(residuals ** 2) / df_resid

            XtX_inv = np.linalg.inv(self.X_train.T @ self.X_train)
            var_betas = s_squared * np.diag(XtX_inv)

            se_betas = np.sqrt(var_betas)
            t_stats = self.coefficients / se_betas

            from scipy.stats import t
            p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=df_resid))

            # Output
            print(f"Hypothesis Test for Feature: {feature_names[idx]}")
            coef_names = ['Intercept', feature_names[idx]]
            print(pd.DataFrame({
                'Coefficient': coef_names,
                'Estimate': self.coefficients,
                'Std Error': se_betas,
                't-statistic': t_stats,
                'p-value': p_values
            }))

    
    
    
    
    
    def interval_estimation(self, X_new=None, alpha=0.05, sigma=None):
        if self.coefficients is None:
            raise ValueError("Model must be fitted before estimating intervals.")

        n, p = self.X_train.shape  # p includes intercept
        df = n - p
        XtX_inv = np.linalg.inv(self.X_train.T @ self.X_train)

        # Calculate critical value
        if sigma is None:
            sigma_hat = np.sqrt(np.sum(self.residuals ** 2) / df)
            crit_value = stats.t.ppf(1 - alpha / 2, df)
        else:
            sigma_hat = sigma
            crit_value = stats.norm.ppf(1 - alpha / 2)

        # Calculate standard errors and margins
        std_errors = np.sqrt(np.diag(XtX_inv)) * sigma_hat
        margins = crit_value * std_errors

        # Confidence intervals for coefficients
        lower_bounds = self.coefficients - margins
        upper_bounds = self.coefficients + margins
        coef_names = ['Intercept'] + [f'X{i}' for i in range(1, p)]
        coef_df = pd.DataFrame({
            'Coefficient': coef_names,
            'Estimate': self.coefficients,
            'Std Error': std_errors,
            f'{100*(1-alpha):.1f}% CI Lower': lower_bounds,
            f'{100*(1-alpha):.1f}% CI Upper': upper_bounds
        })

        # Prediction intervals
        pred_df = None
        if X_new is not None:
            X_new = np.atleast_2d(X_new)
            if X_new.shape[1] == p - 1:
                X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
            elif X_new.shape[1] != p:
                raise ValueError(f"Expected {p - 1} features, got {X_new.shape[1]}")
    
            y_preds = X_new @ self.coefficients
            se_preds = np.sqrt(np.sum(X_new @ XtX_inv * X_new, axis=1)) * sigma_hat
            margin_preds = crit_value * se_preds
            lower_preds = y_preds - margin_preds
            upper_preds = y_preds + margin_preds

            pred_df = pd.DataFrame({
                'Prediction': y_preds,
                f'{100*(1-alpha):.1f}% PI Lower': lower_preds,
                f'{100*(1-alpha):.1f}% PI Upper': upper_preds
            })

        return coef_df, pred_df
    
    
    


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
        ax.set_xlabel("Age")
        ax.set_ylabel("Experience")
        ax.set_zlabel("Income")
        ax.set_title("Regression Plane vs Actual Data Points")
        plt.legend()
        plt.show()
        print("Plotting complete")
        
        
