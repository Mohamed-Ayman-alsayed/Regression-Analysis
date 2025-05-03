
from fit_predict_plot import MultipleLinearRegression
import numpy as np


X = np.random.rand(100, 2)  # 2 features for plotting
y = 3 + 2*X[:,0] + 1.5*X[:,1] + np.random.normal(0, 0.5, 100)

model = MultipleLinearRegression()
model.fit(X, y)

# Get ANOVA table
model.anova()

# Make predictions
new_X = np.array([[0.5, 0.7], [0.2, 0.3]])
print(model.predict(new_X))

# Plot results
model.plot(X, y)