import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes.csv')

data.head()

data.shape

data.describe()

data['Outcome'].value_counts()

data.groupby('Outcome').mean()

# separating the data and label
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome'] # target variable

X

Y

# Standardizing tha data (Range for values of all columns will become same)
scaler = StandardScaler()
scaler.fit(X)

# Standardized data(helps model to train better)
X = scaler.transform(X)

X

### Splitting the data into train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

### Model trainning :
####  1. Using Support Vector Machine 

from sklearn import svm
model_SVM = svm.SVC(kernel='linear')

# training the model with training data
model_SVM.fit(X_train, Y_train)

prediction_svm = model_SVM.predict(X_test)
prediction_svm_accuracy = accuracy_score(prediction_svm, Y_test)

print('Accuracy using Support Vector Machine : ', prediction_svm_accuracy)

#### 2. Using Logistic Regression 

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()

model_LR.fit(X_train, Y_train)

prediction_LR = model_LR.predict(X_test)
prediction_LR_accuracy = accuracy_score(prediction_LR, Y_test)

print('Accuracy using Logistic Regression : ', prediction_LR_accuracy)

#### 3. Using Decision Tree 

from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, Y_train)

prediction_DT = model_DT.predict(X_test)
prediction_DT_accuracy = accuracy_score(prediction_DT, Y_test)

print('Accuracy using Decision Tree : ', prediction_DT_accuracy)

#### 4. Using Random forest

from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=3, criterion='entropy')
model_RF.fit(X_train, Y_train)

prediction_RF = model_RF.predict(X_test)
prediction_RF_accuracy = accuracy_score(prediction_RF, Y_test)

print('Accuracy using Random Forest : ',prediction_RF_accuracy)

#### 5. Using KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model_KNN.fit(X_train, Y_train)

prediction_KNN = model_KNN.predict(X_test)
prediction_KNN_accuracy = accuracy_score(prediction_KNN, Y_test)

print('Accuracy using KNN classifier : ',prediction_KNN_accuracy)

### Making a predictive system

# Accuracy was highest when we used Support Vector Machine so we will use that model for our predictive system

input = (3,158,76,36,245,31.6,0.851,28)

# converting input data into numpy array
input_array = np.asarray(input)

# reshaping the input data as we are predicting for 1 instance
reshaped_input = input_array.reshape(1,-1)

# standardizing the input data
std_data = scaler.transform(reshaped_input)
#print(std_data)

prediction = model_SVM.predict(std_data)
print(prediction)

if(prediction[0]==1):
    print('Person is diabetic')
else:
    print('Person is non-diabetic')
