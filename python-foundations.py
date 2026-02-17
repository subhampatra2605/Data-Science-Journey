"""
Python Foundations for Data Science
Author: Your Name
Purpose: Core Python concepts used in data analysis & ML
"""

# -------------------------
# 1. Variables & Data Types
# -------------------------
x = 10
y = 3.5
name = "Data Science"
is_active = True

print(type(x), type(y), type(name), type(is_active))

# -------------------------
# 2. Type Casting
# -------------------------
a = "100"
b = int(a)
c = float(b)
print(b, c)

# -------------------------
# 3. Operators
# -------------------------
print(10 + 5)     # Arithmetic
print(10 > 5)     # Comparison
print(True and False)  # Logical
print(5 in [1, 2, 3, 4, 5])  # Membership

# -------------------------
# 4. Conditional Statements
# -------------------------
score = 75

if score >= 90:
    result = "Excellent"
elif score >= 60:
    result = "Pass"
else:
    result = "Fail"

print(result)

# -------------------------
# 5. Loops
# -------------------------
for i in range(5):
    print(i)

count = 0
while count < 3:
    print("Looping")
    count += 1

# -------------------------
# 6. Break & Continue
# -------------------------
for i in range(10):
    if i == 5:
        break
    print(i)

# -------------------------
# 7. Strings
# -------------------------
text = "data science"
print(text.upper())
print(text.split())

# -------------------------
# 8. Lists
# -------------------------
numbers = [1, 2, 3, 4]
numbers.append(5)
print(numbers)

# -------------------------
# 9. Tuples
# -------------------------
point = (10, 20)
print(point[0])

# -------------------------
# 10. Dictionaries
# -------------------------
student = {"name": "Alex", "score": 85}
print(student["name"])

# -------------------------
# 11. Sets
# -------------------------
unique_values = {1, 2, 2, 3}
print(unique_values)

# -------------------------
# 12. Functions
# -------------------------
def calculate_average(values):
    return sum(values) / len(values)

print(calculate_average([10, 20, 30]))

# -------------------------
# 13. Lambda Function
# -------------------------
square = lambda x: x ** 2
print(square(5))

# -------------------------
# 14. Modules
# -------------------------
import math
print(math.sqrt(16))

# -------------------------
# 15. Simple Problem Solving
# -------------------------
def is_even(num):
    return num % 2 == 0

print(is_even(10))
