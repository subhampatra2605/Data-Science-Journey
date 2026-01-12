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

# =========================
# SEABORN COMPLETE MASTER FILE
# =========================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------
# 1. LOAD DATASETS
# -------------------------
tips = sns.load_dataset("tips")
titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")

# -------------------------
# 2. BASIC SETTINGS
# -------------------------
sns.set_theme(style="darkgrid")

# -------------------------
# 3. LINE PLOT
# -------------------------
sns.lineplot(x="size", y="total_bill", data=tips)
plt.show()

# -------------------------
# 4. SCATTER PLOT
# -------------------------
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.show()

# -------------------------
# 5. BAR PLOT
# -------------------------
sns.barplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 6. COUNT PLOT
# -------------------------
sns.countplot(x="day", data=tips)
plt.show()

# -------------------------
# 7. HISTOGRAM
# -------------------------
sns.histplot(tips["total_bill"], bins=20, kde=True)
plt.show()

# -------------------------
# 8. BOX PLOT
# -------------------------
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 9. VIOLIN PLOT
# -------------------------
sns.violinplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 10. STRIP PLOT
# -------------------------
sns.stripplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 11. SWARM PLOT
# -------------------------
sns.swarmplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 12. HEATMAP
# -------------------------
corr = iris.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# -------------------------
# 13. PAIR PLOT
# -------------------------
sns.pairplot(iris, hue="species")
plt.show()

# -------------------------
# 14. JOINT PLOT
# -------------------------
sns.jointplot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.show()

# -------------------------
# 15. KDE PLOT
# -------------------------
sns.kdeplot(tips["total_bill"], shade=True)
plt.show()

# -------------------------
# 16. REGRESSION PLOT
# -------------------------
sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()

# -------------------------
# 17. CATPLOT (ALL-IN-ONE)
# -------------------------
sns.catplot(x="day", y="total_bill", data=tips, kind="box")
plt.show()

# -------------------------
# 18. FACET GRID
# -------------------------
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

# -------------------------
# 19. STYLE & PALETTES
# -------------------------
sns.set_style("whitegrid")
sns.set_palette("Set2")

sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

# -------------------------
# 20. SAVE PLOT
# -------------------------
sns.lineplot(x="size", y="tip", data=tips)
plt.savefig("seaborn_plot.png")
plt.show()

# =========================
# END OF SEABORN
# =========================
