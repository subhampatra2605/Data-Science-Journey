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

# =========================
# PANDAS COMPLETE MASTER FILE
# =========================

import pandas as pd
import numpy as np

# -------------------------
# 1. SERIES
# -------------------------
s = pd.Series([10, 20, 30, 40])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# -------------------------
# 2. DATAFRAME CREATION
# -------------------------
data = {
    'Name': ['A', 'B', 'C'],
    'Age': [21, 22, 23],
    'Marks': [85, 90, 88]
}

df = pd.DataFrame(data)

df2 = pd.DataFrame(
    np.array([[1,2],[3,4]]),
    columns=['X', 'Y']
)

# -------------------------
# 3. BASIC INFO
# -------------------------
df.head()
df.tail()
df.shape
df.columns
df.index
df.info()
df.describe()

# -------------------------
# 4. COLUMN & ROW ACCESS
# -------------------------
df['Name']
df[['Name', 'Marks']]

df.loc[0]
df.loc[0, 'Name']

df.iloc[1]
df.iloc[1, 2]

# -------------------------
# 5. ADD / REMOVE COLUMNS
# -------------------------
df['Grade'] = ['A', 'A+', 'A']
df.drop('Grade', axis=1, inplace=False)

# -------------------------
# 6. ADD / REMOVE ROWS
# -------------------------
df.loc[len(df)] = ['D', 24, 92]
df.drop(0, inplace=False)

# -------------------------
# 7. FILTERING DATA
# -------------------------
df[df['Marks'] > 85]
df[(df['Age'] > 21) & (df['Marks'] > 85)]

# -------------------------
# 8. SORTING
# -------------------------
df.sort_values(by='Marks')
df.sort_values(by='Age', ascending=False)

# -------------------------
# 9. GROUPBY
# -------------------------
df.groupby('Age')['Marks'].mean()

# -------------------------
# 10. MISSING VALUES
# -------------------------
df.isnull()
df.isnull().sum()

df.fillna(0)
df.dropna()

# -------------------------
# 11. APPLY & MAP
# -------------------------
df['Marks'].apply(lambda x: x + 5)

df['Grade'] = df['Marks'].map(
    lambda x: 'A' if x >= 90 else 'B'
)

# -------------------------
# 12. MERGE & CONCAT
# -------------------------
df_left = pd.DataFrame({'ID':[1,2], 'Name':['A','B']})
df_right = pd.DataFrame({'ID':[1,2], 'Marks':[80,90]})

pd.merge(df_left, df_right, on='ID')

pd.concat([df_left, df_right], axis=0)

# -------------------------
# 13. FILE HANDLING
# -------------------------
df.to_csv("data.csv", index=False)
pd.read_csv("data.csv")

df.to_excel("data.xlsx", index=False)
pd.read_excel("data.xlsx")

# -------------------------
# 14. DATE & TIME
# -------------------------
dates = pd.date_range("2025-01-01", periods=5)
df_dates = pd.DataFrame({'Date': dates})

df_dates['Year'] = df_dates['Date'].dt.year
df_dates['Month'] = df_dates['Date'].dt.month

# -------------------------
# 15. DUPLICATES
# -------------------------
df.duplicated()
df.drop_duplicates()

# =========================
# END OF PANDAS
# =========================

# =========================
# MATPLOTLIB COMPLETE MASTER FILE
# =========================

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. BASIC LINE PLOT
# -------------------------
x = np.arange(1, 6)
y = x ** 2

plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

# -------------------------
# 2. MULTIPLE LINES
# -------------------------
plt.plot(x, x)
plt.plot(x, x**2)
plt.plot(x, x**3)
plt.show()

# -------------------------
# 3. SCATTER PLOT
# -------------------------
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()

# -------------------------
# 4. BAR CHART
# -------------------------
names = ['A', 'B', 'C']
marks = [85, 90, 88]

plt.bar(names, marks)
plt.title("Bar Chart")
plt.show()

# -------------------------
# 5. HORIZONTAL BAR
# -------------------------
plt.barh(names, marks)
plt.show()

# -------------------------
# 6. HISTOGRAM
# -------------------------
data = np.random.randn(100)

plt.hist(data, bins=10)
plt.title("Histogram")
plt.show()

# -------------------------
# 7. PIE CHART
# -------------------------
plt.pie(marks, labels=names, autopct='%1.1f%%')
plt.title("Pie Chart")
plt.show()

# -------------------------
# 8. SUBPLOTS
# -------------------------
plt.subplot(1, 2, 1)
plt.plot(x, y)

plt.subplot(1, 2, 2)
plt.bar(names, marks)

plt.show()

# -------------------------
# 9. GRID & STYLE
# -------------------------
plt.plot(x, y)
plt.grid(True)
plt.show()

# -------------------------
# 10. SAVE FIGURE
# -------------------------
plt.plot(x, y)
plt.savefig("plot.png")
plt.show()

# -------------------------
# 11. REALISTIC DATA VISUAL
# -------------------------
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
sales = [200, 220, 250, 230, 260]

plt.plot(days, sales)
plt.title("Weekly Sales")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.show()

# =========================
# END OF MATPLOTLIB
# =========================
