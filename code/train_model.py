import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('../data/joint_data_collection.csv')
X = df[['review_text', 'free_cancellation', 'reviews_count_scaled', 'price_scaled']]
y = df['score']

X_train = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/train/training_data.csv').drop('score', axis=1)
y_train = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/train/training_data.csv')['score']
X_test = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/validation/test_data.csv').drop('score', axis=1)
y_test = pd.read_csv('../learningBase_sentiment_analysis/tmp/learningBase/validation/test_data.csv')['score']

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

model = tf.keras.Sequential([
   tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, 
            loss='huber', 
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

print("Model summary:")
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
   monitor='val_loss',
   patience=30,
   restore_best_weights=True
)

history = model.fit(
   X_train, y_train,
   epochs=500,
   batch_size=32,
   validation_data=(X_test, y_test),
   callbacks=[early_stopping]
)

base_path = '../learningBase_sentiment_analysis/tmp/learningBase'
model.save(f'{base_path}/currentAiSolution.h5')

# Training curves visualization
plt.figure(figsize=(20, 5))

# Loss plot
plt.subplot(1, 4, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# MSE plot
plt.subplot(1, 4, 2)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# RMSE plot
plt.subplot(1, 4, 3)
plt.plot(history.history['rmse'], label='Training RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Model RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(1, 4, 4)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{base_path}/training_curves.png')
plt.close()

# Scatter plot
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test, alpha=0.3, color='blue', label='Actual Scores')
plt.scatter(y_test, y_pred, alpha=0.5, color='orange', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', 
        color='green', label='Perfect Prediction', linewidth=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Prediction Scatter Plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{base_path}/scatter_plot.png')
plt.close()

# Residual plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred.flatten()
plt.scatter(y_pred.flatten(), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig(f'{base_path}/residual_plot.png')
plt.close()

# Save metrics
metrics = {
   'training_iterations': len(history.history['loss']),
   'final_loss': float(history.history['loss'][-1]),
   'final_val_loss': float(history.history['val_loss'][-1]),
   'final_mse': float(history.history['mse'][-1]),
   'final_val_mse': float(history.history['val_mse'][-1]),
   'final_rmse': float(history.history['rmse'][-1]),
   'final_val_rmse': float(history.history['val_rmse'][-1]),
   'final_mae': float(history.history['mae'][-1]),
   'final_val_mae': float(history.history['val_mae'][-1]),
   'mse': float(mean_squared_error(y_test, y_pred)),
   'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
   'r2_score': float(r2_score(y_test, y_pred))
}

pd.Series(metrics).to_csv(f'{base_path}/model_metrics.csv')
pd.DataFrame(history.history).to_csv(f'{base_path}/training_history.csv')
