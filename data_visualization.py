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