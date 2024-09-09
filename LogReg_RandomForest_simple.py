# Description: This script is used to simulate EEG data and train a logistic regression model to classify eyes open and eyes closed states.
#%%

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(43)

# Number of 1-second epochs for EO and EC
num_epochs = 1000
size_of_epoch = 200  # Number of time points per epoch

# Generate time-series like data for each epoch with EO and EC segments
# EO State: Lower alpha, higher beta
alpha_EO = [np.random.normal(loc=10, scale=2, size=size_of_epoch) for _ in range(num_epochs)]
beta_EO = [np.random.normal(loc=25, scale=5, size=size_of_epoch) for _ in range(num_epochs)]

# EC State: Higher alpha, lower beta
alpha_EC = [np.random.normal(loc=25, scale=5, size=size_of_epoch) for _ in range(num_epochs)]
beta_EC = [np.random.normal(loc=10, scale=2, size=size_of_epoch) for _ in range(num_epochs)]

# Labels: 0 for EO, 1 for EC
labels_EO = [0] * num_epochs
labels_EC = [1] * num_epochs

# Combine data for EO and EC
alpha_combined = alpha_EO + alpha_EC
beta_combined = beta_EO + beta_EC
labels_combined = labels_EO + labels_EC

# Create a DataFrame where each row contains the alpha, beta time series and the label
df_epochs = pd.DataFrame({
    'alpha_power_timeseries': alpha_combined,
    'beta_power_timeseries': beta_combined,
    'label': labels_combined
})

# Shuffle the DataFrame to mix EO and EC epochs
df_epochs = df_epochs.sample(frac=1).reset_index(drop=True)

df_epochs.head()

# %%
# Plot the simulated data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df_epochs.loc[0, 'alpha_power_timeseries'], label='Alpha Power(EO)')
plt.plot(df_epochs.loc[0, 'beta_power_timeseries'], label='Beta Power(EO)')
plt.plot(df_epochs.loc[1, 'alpha_power_timeseries'], label='Alpha Power(EC)')
plt.plot(df_epochs.loc[1, 'beta_power_timeseries'], label='Beta Power(EC)')
plt.xlabel('Time Points')
plt.ylabel('Power')
plt.title('Simulated EEG Data for EO State')
plt.legend()
plt.show()




# %%
# Feature extraction approach
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Feature extraction function: compute summary statistics for each time series
def extract_features(row):
    features = []
    
    # Extract statistics from alpha and beta power time series
    for series in [row['alpha_power_timeseries'], row['beta_power_timeseries']]:
        features.append(np.mean(series))    # Mean
        features.append(np.std(series))     # Standard deviation
        features.append(np.min(series))     # Minimum
        features.append(np.max(series))     # Maximum
    
    return features

# Apply feature extraction to the DataFrame
df_features = df_epochs.apply(extract_features, axis=1, result_type='expand')

# Rename columns for clarity
df_features.columns = ['alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max', 
                       'beta_mean', 'beta_std', 'beta_min', 'beta_max']

# Add the labels back
df_features['label'] = df_epochs['label']

# Split the data into training and test sets
X = df_features.drop(columns=['label'])
y = df_features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Apply a Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")

# %%
# explore the feature importance for random forest model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# %%
# explore the feature importance for logistic regression model
coefs = log_reg.coef_[0]
indices = np.argsort(np.abs(coefs))[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), coefs[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# %%
# project the data into 2D space using PCA
from sklearn.decomposition import PCA
import seaborn as sns

# Apply PCA to the feature matrix
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame with the PCA-transformed data
df_pca = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'label': y
})

# Plot the data in 2D space
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='label', data=df_pca, palette='viridis')
plt.title('PCA Projection of EEG Data')
plt.show()

#%%
# plot log reg decision boundary to the 2D PCA space
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to the standardized feature matrix
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA-transformed data
df_pca = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'label': y
})

# Fit a logistic regression model to the PCA-transformed data
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_pca, y)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='label', data=df_pca, palette='viridis')

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = log_reg_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.title('PCA Projection with Decision Boundary')
plt.show()

# %%
