# ===============================
# PYTHON PRACTICE FILE
# Covers: Basics → OOP → Modules → JSON → Decorators → DSA basics
# ===============================


# ---------- 01. BASICS ----------
print("Hello, World!")
user_name = "Subham"
print(f"Hi, {user_name}! Welcome to Python learning.")
print("Type of user_name:", type(user_name))


# ---------- 02. VARIABLES & OPERATORS ----------
a, b = 10, 20
a, b = b, a  # swap
print("After swapping:", a, b)
print("Sum:", a + b)
print("Comparison Result:", a > b)
print("Logical Result:", True and False)


# ---------- 03. STRINGS ----------
s = "python"
print("Reversed:", s[::-1])
print("Is palindrome:", s == s[::-1])
print("Vowels count:", sum(ch in "aeiou" for ch in s))
print("Formatted:", "Learning {}!".format(s))


# ---------- 04. LISTS ----------
nums = [1, 2, 2, 3, 4]
nums.append(6)
nums.remove(2)
print("Unique list:", list(set(nums)))
squares = [n*n for n in range(5)]
print("Squares:", squares)


# ---------- 05. TUPLES ----------
t = (1, 2, 3)
x, y, z = t
print("Tuple unpacking:", x, y, z)


# ---------- 06. SETS ----------
s1 = {1, 2, 3}
s2 = {3, 4, 5}
print("Union:", s1 | s2)
print("Intersection:", s1 & s2)


# ---------- 07. DICTIONARIES ----------
sentence = "hello world hello AI"
word_count = {word: sentence.split().count(word) for word in sentence.split()}
print("Word frequency:", word_count)


# ---------- 08. CONDITIONALS ----------
num = 7
if num % 2 == 0:
    print("Even")
else:
    print("Odd")


# ---------- 09. LOOPS ----------
for i in range(1, 6):
    print("Table row:", num * i)

# Pattern
for i in range(1, 6):
    print("*" * i)


# ---------- 10. FUNCTIONS ----------
def add(x, y=10):
    return x + y

print("Addition:", add(5))

# Recursive Fibonacci
def fibonacci(n):
    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

print("Fibonacci(6):", fibonacci(6))

# Lambda function
double = lambda x: x * 2
print("Lambda double:", double(4))


# ---------- 11. MODULE DEMO ----------
import math
print("Square root:", math.sqrt(16))


# ---------- 12. FILE HANDLING ----------
with open("sample.txt", "w") as f:
    f.write("Hello from Python!")

with open("sample.txt", "r") as f:
    print("File content:", f.read())


# ---------- 13. EXCEPTION HANDLING ----------
try:
    res = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero!")


# ---------- 14. OOP ----------
class Car:
    wheels = 4  # Class variable

    def __init__(self, brand, year):
        self.brand = brand
        self.year = year

    def description(self):
        return f"{self.brand} - {self.year}"

car1 = Car("Tesla", 2024)
print(car1.description())

# Inheritance
class EV(Car):
    def battery(self):
        return "Battery: 85 kWh"

ev = EV("Tesla Model Y", 2025)
print(ev.description(), ev.battery())


# ---------- 15. ITERATORS & GENERATORS ----------
def my_generator(n):
    for i in range(n):
        yield i * i

gen = my_generator(5)
print("Generator output:", list(gen))


# ---------- 16. DECORATORS ----------
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@debug
def multiply(a, b):
    return a * b

print("Multiply:", multiply(3, 4))


# ---------- 17. COMPREHENSIONS ----------
numbers = [1, 2, 3, 3, 4]
set_comp = {x for x in numbers}
dict_comp = {x: x * x for x in numbers}
print("Set comp:", set_comp)
print("Dict comp:", dict_comp)


# ---------- 18. JSON ----------
import json

data = {"name": "Subham", "role": "AI Learner"}
json_str = json.dumps(data)
print("JSON:", json_str)

parsed = json.loads(json_str)
print("Parsed JSON:", parsed)


# ---------- 19. RANDOM & DATETIME ----------
import random, datetime
print("Random password:", ''.join(random.choice("abc123!@#") for _ in range(8)))
print("Today:", datetime.date.today())


# ---------- 20. DSA Mini-Exercises ----------
# Stack
stack = []
stack.append(1)
stack.append(2)
print("Stack pop:", stack.pop())

# Queue
from collections import deque
queue = deque([1,2,3])
queue.append(4)
print("Queue pop:", queue.popleft())

# Priority Queue
import heapq
pq = []
heapq.heappush(pq, 5)
heapq.heappush(pq, 1)
heapq.heappush(pq, 3)
print("Smallest in priority queue:", heapq.heappop(pq))


print("\n=========== COMPLETED: PYTHON PRACTICE SHEET ===========")


