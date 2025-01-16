import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


# First I loaded the dataset to be used for data mining.
data = pd.read_csv("data/Testing_carlease.csv")

# I deleted the columns with NA.
data_cleaned = data.dropna(axis=1, how='all')

# After deleting the empty columns, I created a new data set from the remaining ones and named it “data_cleaned”
data_cleaned.to_csv('data_cleaned.csv', index=False)
print("New dataset saved as 'data_cleaned.csv'")

# I uploaded the file to do the data mining in the new file I created by deleting the empty columns.
data_cleaned = pd.read_csv('data_cleaned.csv')

# I checked my newly created file for missing data.
print("Number of missing data in the new dataset:")
print(data_cleaned.isnull().sum())

## I fixed the data ranking (Easy, Medium, Hard)
data_cleaned = data_cleaned.sort_values(by=['Time'])

kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++')
# I fixed the start centers with the code above

# Using KMeans, I divided the “Time” dimension into 3 classes (Easy, Medium, Hard)
data_cleaned['cluster'] = kmeans.fit_predict(data_cleaned[['Time']])

# I used KMeans to sort the cluster centers into 3 separate clusters to determine the most accurate threshold values.
cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

# I classified the values into clusters.
def classify_time(value):
    if value < cluster_centers[1]:
        return 'Easy'  # Corrected 'Esay' to 'Easy'
    elif cluster_centers[1] <= value < cluster_centers[2]:
        return 'Medium'
    else:
        return 'Hard'

data_cleaned['Difficulty'] = data_cleaned['Time'].apply(classify_time)

# I converted the string columns (Difficulty) into numeric data that the VM algorithm can process (0, 1, 2)
label_encoder = LabelEncoder()
data_cleaned['Difficulty'] = label_encoder.fit_transform(data_cleaned['Difficulty'])

# Select numeric columns (columns other than Time and Difficulty)
X = data_cleaned.select_dtypes(include=[np.number]).drop(columns=['Difficulty', 'Time', 'cluster'])
y = data_cleaned['Difficulty']

# I performed a normalization operation.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# I divided it into training and test sets (Training 80%, Test 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# I created and trained a KNN model. By trying “kn” from 3 to 7, I found that it makes the least number of hahas at 3
# that's why I took “kn” as 3.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# I tested the model with test data that I allocated as 20% of my dataset.
y_pred = knn.predict(X_test)

# I printed the accuracy rate according to the result obtained in the test.
accuracy = accuracy_score(y_test, y_pred)

# I printed the classification report.
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# I made charts

# 1. I created and visualized the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Estimated")
plt.ylabel("Actual Value")
plt.show()

# 2. I created and visualized Boxplot - Time column
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_cleaned['Time'])
plt.title("Time Variable Boxplot")
plt.show()

# 3. I created and visualized the Cluster Centers Graph
plt.figure(figsize=(6, 4))
plt.plot(cluster_centers, marker='o', linestyle='--', label="Küme Merkezleri")
plt.title("KMeans Cluster Centers")
plt.xlabel("Clusters")
plt.ylabel("Time Value")
plt.legend()
plt.show()

# 4. I created and visualized Feature Importance Ranking with Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_
features = X.columns

# 5. I visualized the feature importance scores calculated with the Random Forest algorithm as a plot.
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance Ranking (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

#6. ROC
# Convert 'y_test' and 'y_pred' into binary format for multi-class labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Specify the classes (0, 1, 2) here
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])

# Iterate over each class to compute the ROC curve
fpr, tpr, roc_auc = {}, {}, {}
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# General settings for the ROC curve plot
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Multi-Class)')
plt.legend(loc='best')
plt.show()


