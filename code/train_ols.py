import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X_train = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/train/training_data.csv').drop('score', axis=1)
y_train = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/train/training_data.csv')['score']
X_test = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/validation/test_data.csv').drop('score', axis=1)
y_test = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/validation/test_data.csv')['score']

# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit OLS model
model = sm.OLS(y_train, X_train_sm).fit()

# Save model
base_path = '../learningBase_sentiment_analysis/tmp/learningBase'
model.save(f'{base_path}/currentOlsSolution.pkl')

# Make predictions
y_pred = model.predict(X_test_sm)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test, alpha=0.3, color='blue', label='Actual Scores')
plt.scatter(y_test, y_pred, alpha=0.5, color='orange', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', 
        color='green', label='Perfect Prediction', linewidth=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('OLS Prediction Scatter Plot')
plt.legend()
plt.grid(True)
plt.savefig(f'{base_path}/ols_scatter_plot.png')
plt.close()

# Residual plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('OLS Residual Plot')
plt.grid(True)
plt.savefig(f'{base_path}/ols_residual_plot.png')
plt.close()

# Save metrics
metrics = {
   'r2_score': float(r2_score(y_test, y_pred)),
   'mse': float(mean_squared_error(y_test, y_pred)),
   'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
   'model_summary': model.summary().as_text()
}

pd.Series(metrics).to_csv(f'{base_path}/ols_metrics.csv')

# Compare metrics
nn_metrics = pd.read_csv(f'{base_path}/model_metrics.csv', index_col=0)
ols_metrics = pd.read_csv(f'{base_path}/ols_metrics.csv', index_col=0)

print("\nModel Performance Comparison:")
print(f"Neural Network R2: {float(nn_metrics.loc['r2_score'].values[0]):.4f}")
print(f"OLS R2: {metrics['r2_score']:.4f}")  # Use directly from metrics dict
print(f"Neural Network RMSE: {float(nn_metrics.loc['rmse'].values[0]):.4f}")
print(f"OLS RMSE: {metrics['rmse']:.4f}")

print("\nOLS Model Summary:")
print(model.summary())
