"""
Mathematics for Data Science
Author: Your Name
Purpose: Statistics & math concepts used in ML & analytics
"""

import numpy as np
from scipy import stats

# -------------------------
# 1. Population vs Sample
# -------------------------
population = np.array([10, 20, 30, 40, 50])
sample = np.array([20, 30, 40])

# -------------------------
# 2. Measure of Central Tendency
# -------------------------
print("Mean:", np.mean(sample))
print("Median:", np.median(sample))
print("Mode:", stats.mode(sample, keepdims=True)[0])

# -------------------------
# 3. Measure of Variability
# -------------------------
print("Variance:", np.var(sample))
print("Standard Deviation:", np.std(sample))

# -------------------------
# 4. Percentiles & Quartiles
# -------------------------
print("25th Percentile:", np.percentile(sample, 25))
print("75th Percentile:", np.percentile(sample, 75))

# -------------------------
# 5. Probability
# -------------------------
probability_event = 2 / 6
print("Probability:", probability_event)

# -------------------------
# 6. Normal Distribution
# -------------------------
data = np.random.normal(50, 10, 1000)
print("Mean:", np.mean(data))
print("Std Dev:", np.std(data))

# -------------------------
# 7. Covariance & Correlation
# -------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

print("Covariance:", np.cov(x, y)[0][1])
print("Correlation:", np.corrcoef(x, y)[0][1])

# -------------------------
# 8. Central Limit Theorem (Concept Demo)
# -------------------------
sample_means = []
for _ in range(1000):
    sample = np.random.choice(data, size=30)
    sample_means.append(np.mean(sample))

print("CLT Mean:", np.mean(sample_means))

# -------------------------
# 9. Hypothesis Testing (T-Test)
# -------------------------
group1 = np.random.normal(50, 5, 30)
group2 = np.random.normal(55, 5, 30)

t_stat, p_value = stats.ttest_ind(group1, group2)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# -------------------------
# 10. Z-Test (Manual Explanation)
# -------------------------
z_score = (np.mean(group1) - 50) / (np.std(group1) / np.sqrt(len(group1)))
print("Z-Score:", z_score)

# -------------------------
# 11. Outliers using IQR
# -------------------------
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Outlier bounds:", lower_bound, upper_bound)

# -------------------------
# 12. Feature Scaling
# -------------------------
scaled_data = (data - np.mean(data)) / np.std(data)
print("Scaled mean:", np.mean(scaled_data))

# -------------------------
# 13. Cost Function (MSE)
# -------------------------
actual = np.array([3, 5, 7])
predicted = np.array([2.5, 5.1, 6.8])

mse = np.mean((actual - predicted) ** 2)
print("Mean Squared Error:", mse)
