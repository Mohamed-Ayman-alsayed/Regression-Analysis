from MultipleLinearRegression import MultipleLinearRegression
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('Income.csv')  

# Separate features (X) and target (y)
X = df.drop(columns=['Income']).values  # drop Income column to get features
y = df['Income'].values  # Income column is the target

# Prepare feature names 
feature_names = df.drop(columns=['Income']).columns.tolist()

# Create the model
model = MultipleLinearRegression()
model.fit(X, y)

# Fit the model on ALL features
model.fit(X, y)

# Predict (using training data)
y_pred = model.predict(X)

# Evaluate the full model
print("\nOverall Model ANOVA Table:")
anova_table = model.anova(X, y, feature_names)  # Feature-by-feature modified version

print("\nOverall Model Hypothesis Test Table:")
hypothesis_table = model.hypothesis_test(X, y, feature_names)

# # Make predictions
# new_X = np.array([[0.5, 0.7], [0.2, 0.3]])
# print(model.predict(new_X))

# Plot results
model.plot(X, y)