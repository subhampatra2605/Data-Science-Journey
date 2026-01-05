# =========================
# NUMPY COMPLETE MASTER FILE
# =========================

import numpy as np

# -------------------------
# 1. ARRAY CREATION
# -------------------------
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

np.zeros((2, 3))
np.ones((2, 3))
np.full((2, 2), 9)
np.eye(3)

np.arange(0, 10, 2)
np.linspace(0, 1, 5)

np.random.rand(2, 2)
np.random.randn(2, 2)
np.random.randint(1, 100, size=(2, 3))

# -------------------------
# 2. ARRAY PROPERTIES
# -------------------------
a.shape
a.ndim
a.size
a.dtype

# -------------------------
# 3. RESHAPING & STRUCTURE
# -------------------------
x = np.arange(12)

x.reshape(3, 4)
x.reshape(-1, 2)
x.flatten()
x.ravel()
x.T

# -------------------------
# 4. INDEXING & SLICING
# -------------------------
arr = np.array([10, 20, 30, 40, 50])

arr[0]
arr[-1]
arr[1:4]

mat = np.array([[1,2,3],[4,5,6]])

mat[0, 1]
mat[:, 1]
mat[1, :]
mat[0:2, 1:3]

# -------------------------
# 5. BOOLEAN INDEXING
# -------------------------
arr[arr > 25]
arr[arr % 2 == 0]

np.where(arr > 30, arr, 0)

# -------------------------
# 6. MATHEMATICAL OPERATIONS
# -------------------------
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b
a - b
a * b
a / b

np.sqrt(a)
np.exp(a)
np.log(a)
np.sin(a)
np.cos(a)

# -------------------------
# 7. AGGREGATION FUNCTIONS
# -------------------------
data = np.array([[1,2,3],[4,5,6]])

np.sum(data)
np.mean(data)
np.median(data)
np.std(data)
np.var(data)

np.sum(data, axis=0)
np.sum(data, axis=1)

np.min(data)
np.max(data)
np.argmin(data)
np.argmax(data)

# -------------------------
# 8. COMPARISON & LOGICAL
# -------------------------
data > 3
data == 5

np.any(data > 5)
np.all(data > 0)

# -------------------------
# 9. SORTING & SEARCHING
# -------------------------
arr = np.array([40, 10, 30, 20])

np.sort(arr)
np.argsort(arr)

np.unique(arr)
np.in1d(arr, [10, 50])

# -------------------------
# 10. STACKING & SPLITTING
# -------------------------
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate((a, b))
np.vstack((a, b))
np.hstack((a, b))

np.split(b, 3)

# -------------------------
# 11. COPY vs VIEW
# -------------------------
a = np.array([1, 2, 3])
b = a.copy()
c = a.view()

# -------------------------
# 12. BROADCASTING
# -------------------------
a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2,3])

a + b

# -------------------------
# 13. LINEAR ALGEBRA
# -------------------------
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

np.dot(A, B)
A @ B

np.linalg.det(A)
np.linalg.inv(A)
np.linalg.eig(A)

# -------------------------
# 14. STATISTICS
# -------------------------
data = np.array([10, 20, 30, 40, 50])

np.mean(data)
np.std(data)
np.percentile(data, 50)
np.corrcoef(data, data)

# -------------------------
# 15. FILE HANDLING
# -------------------------
np.save("array.npy", data)
np.load("array.npy")

np.savetxt("array.txt", data)
np.loadtxt("array.txt")

# -------------------------
# 16. NAN & INF HANDLING
# -------------------------
arr = np.array([1, 2, np.nan, np.inf])

np.isnan(arr)
np.isinf(arr)
np.nan_to_num(arr)

# =========================
# END OF NUMPY
# =========================


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
