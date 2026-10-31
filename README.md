# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm (4 Steps)

1. Import the required Python libraries and load the student placement dataset into Google Colab.

2. Separate the input features (CGPA, IQ) and output variable (Placement), then split the dataset into training and testing data.

3. Create and train the Logistic Regression model using the training dataset.

4. Predict the placement status of students and evaluate the model using accuracy score and confusion matrix.

## Program:
~~~
Developed by: S Madhumitha 
RegisterNumber: 212225040217
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv('/content/Placement_Data.csv')
data = data.drop(['sl_no', 'salary'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('status_Placed', axis=1)
y = data['status_Placed']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
result = model.predict(X.iloc[[0]])
if result[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")
~~~


## Output:
<img width="1920" height="1020" alt="Screenshot 2026-10-30 085648" src="https://github.com/user-attachments/assets/70f43b3e-8998-4acf-a1d3-f05ad5d4f59c" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
