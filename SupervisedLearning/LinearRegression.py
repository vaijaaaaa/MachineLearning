import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(r"C:\Users\vaijnath\Desktop\MachineLearning\SupervisedLearning\student_scores.csv")
print("Dataset: ")
print(data)

X = data['Hours'].values
Y = data['Marks'].values

m = 0
b = 0
lr = 0.01
epoches = 1000
n = len(X)

for _ in range(epoches):
    Y_pred = m * X + b
    error = Y - Y_pred
    dm = (-2/n) * sum(X * error)
    db = (-2/n) * sum(error)
    m -= lr * dm
    b -= lr * db

Y_final = m*X+b


plt.scatter(X,Y,color = 'blue',label = "Actual Data")
plt.plot(X,Y_final,color = "red", label = f'Prediction: y = {m: .2f}x + {b:.2f}')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.grid(True)
plt.show()

hours = 7.5
predicted_score = m * hours + b
print(f"\n📚 Predicted score for studying {hours} hours: {predicted_score:.2f}")
