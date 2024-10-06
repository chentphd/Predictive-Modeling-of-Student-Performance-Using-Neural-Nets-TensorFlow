import numpy as np 
import pandas as pd 
# 1. Load in the data
df = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\Predictive Modeling of Student Performance\Predict Students' Dropout and Academic Success UCI Machine Learning.csv", sep=';')
df.head()

#2. Train Test Split 
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size= 0.2)

len(df)
len(train_data)
len(test_data)

#2a. Check if the samples are represetative 
train_data['Target'].value_counts() / len(train_data)
test_data['Target'].value_counts() / len(test_data)


#2c. Train Test Split using Stratified Sampling (Optional) 
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2)
for train_ids, test_ids in split.split(df, df['Target']):
    train_data_2 = df.loc[train_ids]
    test_data_2 = df.loc[test_ids]

train_data_2['Target'].value_counts() / len(train_data)
test_data_2['Target'].value_counts() / len(test_data)

df.iloc[:,0:-1]
#3. X_train and Y_train 
X_train = train_data.iloc[:,0:-1]
y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:,0:-1]
y_test = test_data.iloc[:,-1]

len(train_data)
len(test_data)
len(X_train)
len(y_train)
len(X_test)
len(y_test)

#4.  KNN 
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier( n_neighbors= 3)
knn_model.fit(X_train,y_train)


#5. Knn Prediction
y_pred_KNN = knn_model.predict(X_test)

#6. Knn Results 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_KNN))


#7. Naive Bayes 
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train,y_train)
y_pred_NB = nb_model.predict(X_test)
print(classification_report(y_test, y_pred_NB))

#8. Logistic Regression
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train,y_train)
y_pred_LG = lg_model.predict(X_test)
print(classification_report(y_test, y_pred_LG))

#9. SVM 
from sklearn.svm import SVC 
svc_model = SVC()
svc_model = svc_model.fit(X_train, y_train)

y_pred_SVM = svc_model.predict(X_train)
y_pred_SVM
set(y_pred_SVM)
print(classification_report(y_test, y_pred_SVM))


#10. Random Forest 
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier() 
rf_model = rf_model.fit(X_train,y_train)

y_pred_RF = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_RF))



#11. Neural Net 
import tensorflow as tf 