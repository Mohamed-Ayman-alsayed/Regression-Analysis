# 🧮 Multiple Linear Regression from Scratch in Python

This repository showcases a complete, from-scratch implementation of **Multiple Linear Regression**, built collaboratively by a team of five students. Unlike typical black-box machine learning libraries, this project was designed to demystify the **mathematics**, **statistics**, and **computational logic** behind regression models.

It integrates not only model fitting but also critical evaluation techniques used in research and industry: **ANOVA**, **hypothesis testing**, **interval estimation**, and **3D visualization**.

---

## 🧠 Why This Project?

In a world where plug-and-play ML libraries dominate, it's easy to forget *how* these models work. We wanted to change that. By implementing the regression engine ourselves, we learned:

- What happens when you add an intercept term
- How to compute coefficients using matrix operations
- How residuals inform us about model accuracy
- How to statistically validate model parameters
- How to build confidence/prediction intervals without scikit-learn

This project gave us a solid foundation to understand model assumptions, limitations, and mathematical depth.

---

## ⚙️ Key Features

| Feature                         | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| 📈 **Model Training**           | Fit a multiple linear regression model using the **Normal Equation**       |
| 📊 **ANOVA**                    | Compute ANOVA table with F-statistics and p-values for model significance  |
| 🔬 **Hypothesis Testing**       | Perform t-tests on coefficients to check if each feature significantly contributes |
| 📏 **Interval Estimation**      | Construct confidence & prediction intervals for new predictions            |
| 🌐 **3D Visualization**         | Render a regression plane vs actual data (only for 2-feature models)       |

---

## 📎 Modules & Implementation

### ✅ `fit(X, y)`

Fits the model using:

- Intercept term (added manually)
- Normal Equation:  **(XᵀX)⁻¹Xᵀy**

### ✅ `predict(X)`

Predicts new values for unseen features using the fitted coefficients.

### ✅ `anova()`

Generates an ANOVA table to evaluate the significance of the overall model.

### ✅ `hypothesis_test()`

Uses **t-distribution** to compute p-values for each coefficient, testing if it's statistically different from 0.

### ✅ `interval_estimation(X_new)`

Computes:

- **Confidence intervals** for coefficients
- **Prediction intervals** for new data points

### ✅ `plot(X, y)`

3D visualization using `matplotlib`, displaying regression plane vs real data.

---

## 👥 Team Contributions

| 👤 **Member**       | ⚙️ **Role & Contribution**                                                                 | 🔗 **Profile**                                 |
|--------------------|---------------------------------------------------------------------------------------------|------------------------------------------------|
| 🧠 **Mohamed Hossam**  | Implemented **Simple**, **Multiple**, **Polynomial**, and **Logistic Regression** models     | [LinkedIn](https://www.linkedin.com/in/yourprofile) |
| 🔍 **Teammate 1**      | Built the **K-Nearest Neighbors (KNN)** model and applied **K-Fold Cross-Validation**       | [LinkedIn](https://www.linkedin.com/in/teammate1)   |
| 🤖 **Teammate 2**      | Developed and tuned **Support Vector Machine (SVM)** and **Neural Network (NN)** models     | [LinkedIn](https://www.linkedin.com/in/teammate2)   |
| 📊 **Teammate 3**      | Designed advanced **visualizations**, conducted **error analysis**, and refined **docs**    | [LinkedIn](https://www.linkedin.com/in/teammate3)   |
| 🧹 **Teammate 4**      | Led **data preprocessing**, handled **model evaluation**, and ensured **code integrity**     | [LinkedIn](https://www.linkedin.com/in/teammate4)   |
