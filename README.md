# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![DL 01](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/60a424cf-58a6-41a5-a6c4-1339f1828829)



## DESIGN STEPS

### STEP 1:

Loading the dataset.

### STEP 2:

Split the dataset into training and testing.

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot.

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: VIJAYARAJ V
### Register Number: 212222230174
### Dependencies:
```py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
### Data From Sheets:
```py
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DATA').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
### Data Visualization:
```py
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
x=df[['INPUT']].values
y=df[['OUTPUT']].values
```
### Data split and Preprocessing:
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
```
### Regressive Model:
```py
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 1000)
```
### Loss Calculation:
```py
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
### Evaluate the model:
```py
x_test1 = scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
```
### Prediction:
```py
x_n1 = [[5]]
x_n1_1 = scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
### Dataset Information
![image](https://github.com/vijayarajv1704/basic-nn-model/assets/121303741/97d5d5c3-4d64-432f-87e9-576528a4e0da)



## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/vijayarajv1704/basic-nn-model/assets/121303741/0d5d70fc-45f4-4727-ae3a-cf273e475bf1)


### Training
![image](https://github.com/vijayarajv1704/basic-nn-model/assets/121303741/81ec1ea5-17ee-48c0-9d3f-190ff7ad9b4a)


### Test Data Root Mean Squared Error
![image](https://github.com/vijayarajv1704/basic-nn-model/assets/121303741/d1a652b0-d0db-45a4-a5e3-d4ece319f1d1)



### New Sample Data Prediction
![image](https://github.com/vijayarajv1704/basic-nn-model/assets/121303741/9b63415c-de13-44b5-9aba-6ac44e738083)




## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.

