from sklearn.ensemble import IsolationForest
from numpy import where, quantile
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
x = pd.read_excel('IsolationForestData.xlsx')

# Drop missing values
x.dropna(inplace=True)

# Fit Isolation Forest
iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
pred = iforest.fit_predict(x)

# Get anomaly indices
anom_index = where(pred == -1)[0]  # Extract indices from tuple
values = x.iloc[anom_index]  # Use iloc for row selection

# Plot results
plt.scatter(x.iloc[:, 0], x.iloc[:, 1])
plt.scatter(values.iloc[:, 0], values.iloc[:, 1], color="r")
plt.show()

# Refit Isolation Forest with automatic contamination
iforest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)

# Fit the model
iforest.fit(x)

# Get anomaly scores
scores = iforest.score_samples(x)

# Compute threshold for anomalies
thresh = quantile(scores, 0.02)
print(f"Threshold for anomalies: {thresh}")

# Get new anomaly indices
index = where(scores <= thresh)[0]
values = x.iloc[index]  # Use iloc for row selection

# Plot results again
plt.scatter(x.iloc[:, 0], x.iloc[:, 1])
plt.scatter(values.iloc[:, 0], values.iloc[:, 1], color="r")
plt.show()