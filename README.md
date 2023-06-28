# Salary Prediction using K Nearest Neighbors

This project aims to predict the income of individuals using the K Nearest Neighbors (KNN) algorithm. The prediction is based on various attributes such as age, education level, capital gain, hours worked per week, and the income category.

## Dataset
The dataset used for this project contains information about individuals, including their demographic and employment-related attributes. The attributes used for prediction are as follows:

- **age**: The age of the individual.
- **education.num**: The number of years of education completed by the individual.
- **capital.gain**: The capital gains of the individual.
- **hours.per.week**: The number of hours worked per week by the individual.
- **income**: The income category of the individual, which serves as the target variable.

## Installation
To run the code and reproduce the salary prediction using KNN, follow these steps:

1. Clone this repository: `git clone https://github.com/your_username/salary-prediction.git`
2. Navigate to the project directory: `cd salary-prediction`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the salary prediction script: `python predict_salary.py`

## Usage
To use the salary prediction model in your own project, you can follow these steps:

1. Import the necessary libraries:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
```

2. Load the dataset:
```python
data = pd.read_csv('salary_data.csv')
```

3. Split the dataset into features (X) and target variable (y):
```python
X = data[['age', 'education.num', 'capital.gain', 'hours.per.week']]
y = data['income']
```

4. Split the data into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Preprocess the data by scaling the features:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

6. Create and train the KNN model:
```python
k = 5  # Number of neighbors to consider
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
```

7. Predict the income category for new data:
```python
new_data = [[45, 12, 0, 67]]  # Example new data
new_data = scaler.transform(new_data)  # Scale


For reference also check the python notebook
