
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

file_path = 'D:\codesoft internship\datascience\Advertising.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.describe())


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# Scatter plot to visualize Sales vs TV, Radio, Newspaper
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=1, kind='scatter')
plt.show()

# Step 4: Feature Engineering - Create interaction terms
df['TV_Radio_Interaction'] = df['TV'] * df['Radio']
df['TV_Newspaper_Interaction'] = df['TV'] * df['Newspaper']

# Step 5: Define Features and Target
X = df.drop(columns=['Sales'])  # Features (now include interaction terms)
y = df['Sales']  # Target variable (Sales)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Selection & Training
# Initialize different regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    print(f"{name} Results: RMSE = {rmse}, MAE = {mae}, R² = {r2}")

# Step 8: Hyperparameter Tuning (Random Forest Example)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and performance
print("Best Hyperparameters for Random Forest:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
r2_best_rf = r2_score(y_test, y_pred_best_rf)
print(f"Best Random Forest: RMSE = {rmse_best_rf}, R² = {r2_best_rf}")

# Step 9: Feature Importance (Random Forest)
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Feature Importance from Random Forest')
plt.show()

# Step 10: Visualize the best model predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best_rf, color='purple', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='orange')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales - Best Random Forest Model')
plt.show()

