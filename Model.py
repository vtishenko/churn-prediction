import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import r2_score , classification_report, confusion_matrix, accuracy_score

df = pd.DataFrame()

## Load Data
df = pd.read_csv('train data for churn analysis v2.csv')
df = df.dropna(how="any", axis=0)

## Organizing The Data
df = pd.get_dummies(data=df, columns=['IDENTITY_NAME', 'LAST_PLAN_NAME', 'BILLING_CYCLE_FIXED', 'DIRECTORY_TYPE'], drop_first=True)
x = df.drop(['CHURNED'], axis=1)
y = df['CHURNED']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

## train, test , validate
reg = LogisticRegression( max_iter=1000)
reg.fit(x_train, y_train)
predictions = reg.predict(x_test)


## Tests
tests = [classification_report, confusion_matrix, accuracy_score]
while tests :
    test_name = tests.pop()
    print('{} test '.format(test_name.__name__))
    if test_name.__name__ == 'confusion_matrix':
        print(pd.DataFrame(test_name(y_test, predictions), columns=['True Churn', 'True Not Churn'], index=['Predicted Churn', 'Predicted Not Churn']))
    else:
        print(test_name(y_test, predictions))
    print('*' * 55)
#print("classification_report is {}".format(classification_report(y_test, predictions)))

