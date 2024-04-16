import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"StressLevelDataset.csv")

# Extract features (X) and target variable (y)
X = dataFrame.drop("stress_level", axis=1)
y = dataFrame["stress_level"]

# Train-validation-test split (60:20:20))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
cm = confusion_matrix(y_val, knn1.predict(X_val))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for KNN (k=1)")
plt.show()
knn1_accuracy = accuracy_score(y_val, knn1.predict(X_val))
knn1_precision = precision_score(y_val, knn1.predict(X_val), average="weighted")
knn1_recall = recall_score(y_val, knn1.predict(X_val), average="weighted")
knn1_f1 = f1_score(y_val, knn1.predict(X_val), average="weighted")

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
cm = confusion_matrix(y_val, knn3.predict(X_val))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for KNN (k=3)")
plt.show()
knn3_accuracy = accuracy_score(y_val, knn3.predict(X_val))
knn3_precision = precision_score(y_val, knn3.predict(X_val), average="weighted")
knn3_recall = recall_score(y_val, knn3.predict(X_val), average="weighted")
knn3_f1 = f1_score(y_val, knn3.predict(X_val), average="weighted")

knn_comparison = pd.DataFrame(
    {
        "k": [1, 3],
        "Accuracy": [knn1_accuracy, knn3_accuracy],
        "Precision": [knn1_precision, knn3_precision],
        "Recall": [knn1_recall, knn3_recall],
        "F1": [knn1_f1, knn3_f1],
    }
)
print("-------------------------- KNN --------------------------")
print(knn_comparison)

k = range(1, 21)
training_error = []
validation_error = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    training_error.append(1 - knn.score(X_train, y_train))
    validation_error.append(1 - knn.score(X_val, y_val))

plt.plot(k, training_error, label="Training Error")
plt.plot(k, validation_error, label="Validation Error")
plt.xlabel("k")
plt.ylabel("Error")
plt.title("Training Error vs. Validation Error for KNN")
plt.legend()
plt.show()

hyperparameters = {
    "n_estimators": [100, 200],
    "max_depth": [1, 2],
}
rf_model = RandomForestClassifier(random_state=42)
rf_model = GridSearchCV(rf_model, hyperparameters, cv=5, n_jobs=-1, scoring="accuracy")
rf_model.fit(X_train, y_train)

rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
rf_precision = precision_score(y_val, rf_model.predict(X_val), average="weighted")
rf_recall = recall_score(y_val, rf_model.predict(X_val), average="weighted")
rf_f1 = f1_score(y_val, rf_model.predict(X_val), average="weighted")

rf_results = pd.DataFrame(
    {
        "Accuracy": [rf_accuracy],
        "Precision": [rf_precision],
        "Recall": [rf_recall],
        "F1": [rf_f1],
    }
)
print("-------------------------- Random Forest --------------------------")
print("Best parameters:", rf_model.best_params_)
print("Best accuracy score:", rf_model.best_score_)
cm = confusion_matrix(y_val, rf_model.predict(X_val))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest")
plt.show()
print(rf_results)


hyperparameters = {
    "C": [0.1, 1, 10],
    "max_iter": [100, 200],
}

lr = LogisticRegression(random_state=42)
lr = GridSearchCV(lr, hyperparameters, cv=5, n_jobs=-1, scoring="accuracy")

lr.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_val, lr.predict(X_val))
lr_precision = precision_score(y_val, lr.predict(X_val), average="weighted")
lr_recall = recall_score(y_val, lr.predict(X_val), average="weighted")
lr_f1 = f1_score(y_val, lr.predict(X_val), average="weighted")
cm = confusion_matrix(y_val, lr.predict(X_val))

lr_results = pd.DataFrame(
    {
        "Accuracy": [lr_accuracy],
        "Precision": [lr_precision],
        "Recall": [lr_recall],
        "F1": [lr_f1],
    }
)

print("-------------------------- Logistic Regression --------------------------")
print("Best parameters:", lr.best_params_)
print("Best accuracy score:", lr.best_score_)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
print(lr_results)


print("-------------------------- Best Model --------------------------")
best_model = rf_model
best_params = rf_model.best_params_

# Predict stress levels on the scaled test data using the best model
y_pred_best_model_test = best_model.predict(X_test)

# Evaluate the best model on test data
accuracy_best_model_test = accuracy_score(y_test, y_pred_best_model_test)
precision_best_model_test = precision_score(
    y_test, y_pred_best_model_test, average="weighted"
)
recall_best_model_test = recall_score(
    y_test, y_pred_best_model_test, average="weighted"
)
f1_best_model_test = f1_score(y_test, y_pred_best_model_test, average="weighted")

cm = confusion_matrix(y_test, y_pred_best_model_test)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for Best Model")
plt.show()

print("Accuracy in Best Model ", accuracy_best_model_test)
print("Precision in Best Model ", precision_best_model_test)
print("Recall in Best Model ", recall_best_model_test)
print("F1 Score in Best Model ", f1_best_model_test)

# Print classification error for best model
misclassified_best_model_test = (
    cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[1, 2] + cm[2, 0] + cm[2, 1]
)
print(
    f"Classification Error for Best Model: {misclassified_best_model_test / y_test.shape[0]:.2%}"
)
