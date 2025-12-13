import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ar = np.array([9, 6, 4, 2, 6, 1, 6, 3])

# mean = float(np.sum(ar)/len(ar))

dataset = pd.read_csv('Titanic-Dataset.csv')

dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True) 

md = np.median(dataset['Fare'])

mn = np.mean(dataset['Fare'])

mode = dataset['Fare'].mode()[0]

# print(mode)

count = dataset['Fare'].value_counts()

# print(count)

sns.histplot(data=dataset, x="Fare", bins = [i for i in range(0, 81, 10)])
plt.plot([mn for i in range(0, 300)], [i for i in range(0, 300)], color='red', label='Mean')
plt.plot([md for i in range(0, 300)], [i for i in range(0, 300)], color='blue', label='Median')
plt.plot([mode for i in range(0, 300)], [i for i in range(0, 300)], color='green', label='Mode')
plt.legend()
plt.show()