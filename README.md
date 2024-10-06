
---

# Predictive Modeling of Student Performance

This project applies machine learning models to predict student dropout and academic success using data from the UCI Machine Learning Repository. The dataset contains features related to student demographics, performance, and behavior. Various algorithms such as K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Random Forest, and Neural Networks are used to model student performance and predict outcomes.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Project Overview

The goal of this project is to predict whether a student will drop out, graduate, or stay enrolled using a variety of classification algorithms. The following steps are carried out in the project:

1. Data loading and preprocessing.
2. Train-Test split with and without stratified sampling.
3. Model training with multiple classifiers.
4. Model evaluation using classification reports, confusion matrices, and ROC curves.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/tonychen/predictive_modeling_student_performance.git
    ```
2. Navigate to the project folder:
    ```bash
    cd predictive_modeling_student_performance
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. You will need to download the dataset from the UCI Machine Learning Repository and save it in the appropriate directory.

## Dataset

The dataset used in this project is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Predict+Students%27+Dropout+and+Academic+Success).

### Features

The dataset contains several features such as:

- Demographics
- Academic history
- Behavioral data

### Target

The target variable `Target` has three classes:

1. Graduate
2. Dropout
3. Enrolled

## Models Implemented

The following models were used in this project:

1. **K-Nearest Neighbors (KNN)**:
   - Classifies based on the majority class among the nearest neighbors.
   
2. **Naive Bayes**:
   - A probabilistic classifier based on Bayes' Theorem.

3. **Logistic Regression**:
   - A linear model used for classification tasks.

4. **Support Vector Machines (SVM)**:
   - A powerful classifier that finds the optimal boundary between classes.

5. **Random Forest**:
   - An ensemble learning method based on decision trees.

6. **Neural Networks**:
   - A multi-layer feedforward neural network was built using TensorFlow/Keras.

## Evaluation

Each model was evaluated based on:

- **Accuracy**: The overall correctness of the model.
- **Confusion Matrix**: To visualize misclassifications.
- **Classification Report**: Precision, recall, and F1-score.
- **ROC Curve**: To analyze performance across different thresholds.

## Results

The following are the highlights of the model performances:

- **KNN**:
  - Moderate performance with classification accuracy.
  
- **Naive Bayes**:
  - Fast and effective but less accurate on this dataset.

- **Logistic Regression**:
  - Provided reasonable accuracy with a simple model.

- **SVM**:
  - Good performance but computationally expensive.

- **Random Forest**:
  - High accuracy and good generalization.

- **Neural Networks**:
  - Achieved the highest accuracy after hyperparameter tuning and model training.

## How to Run

You can run the models by executing the `Predictive_Modeling_of_Student_Performance.py` script:

```bash
python Predictive_Modeling_of_Student_Performance.py
```

This will load the dataset, train models, and display evaluation metrics such as confusion matrices, ROC curves, and classification reports.

## Contributions

- **Tony Chen**: Model implementation, data preprocessing, and evaluation.

## License

This project is licensed under the MIT License.

---

