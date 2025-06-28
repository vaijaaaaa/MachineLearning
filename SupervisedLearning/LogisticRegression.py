import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("pass_fail.csv")
X = data['Hours'].values
Y = data['Result'].values


m = 0.0
b = 0.0
lr = 0.01
epochs = 1000
n = len(X)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for _ in range(epochs):
    z = m * X + b
    Y_pred = sigmoid(z)
    
    error = Y_pred - Y
    
    dm = (1/n) * np.dot(X, error)
    db = (1/n) * np.sum(error)
    
    m -= lr * dm
    b -= lr * db


z_final = m * X + b
Y_final = sigmoid(z_final)


plt.scatter(X, Y, label='Actual', color='blue')
plt.plot(X, Y_final, label='Sigmoid Curve', color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression from Scratch")
plt.legend()
plt.grid(True)
plt.show()

# Predict for 7.5 hours
hours = 4
prob = sigmoid(m * hours + b)
result = 1 if prob >= 0.5 else 0

print(f"\n📚 Predicted probability for {hours} hours: {prob:.4f}")
print(f"🎓 Predicted class (0=Fail, 1=Pass): {result}")
