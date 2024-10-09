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
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
df = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\Predictive Modeling of Student Performance\Predict Students' Dropout and Academic Success UCI Machine Learning.csv", sep=';')
df.head()

# Check for missing values
print(df.isnull().sum())

# Since there are no missing values based on your dataset description, we can proceed.

# Convert the target column 'Target' to numerical values using Label Encoding
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

# Define the features (X) and target (y)
X = df.drop(columns=['Target'])
y = df['Target']

# Normalize the numerical columns (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the target variable to categorical (for multi-class classification)
y_categorical = to_categorical(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)
#####################################################
# Build the neural network model
model = Sequential()

# Input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output layer (since we have 3 classes: Graduate, Dropout, or Enrolled)
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
###############
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(f'Test accuracy: {test_acc:.4f}')
###############
# Predict on new data (example from X_test)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Decode the predicted classes to original labels
predicted_labels = le.inverse_transform(predicted_classes)

print(predicted_labels[:5])  # Print first 5 predictions
#################
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class indices
y_true = np.argmax(y_test, axis=1)  # Convert one-hot to class indices

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(y_true, y_pred_classes))


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the output classes for ROC curve
y_test_binarized = label_binarize(y_true, classes=[0, 1, 2])
y_pred_binarized = label_binarize(y_pred_classes, classes=[0, 1, 2])

# Plot ROC curve for each class
for i in range(y_test_binarized.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (area = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
