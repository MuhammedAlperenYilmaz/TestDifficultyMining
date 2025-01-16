# DATA MINING APPLICATIONS ON SOFTWARE TEST DATA: CLASSIFYING DATA ON THE ESTIMATED MAXIMUM TIME A GIVEN TEST TASK WILL TAKE DURING THE EXECUTION PHASE

This project involves performing data mining and machine learning tasks using the "Testing_carlease.csv" dataset. The goal is to classify the "Time" column into three categories: "Easy", "Medium", and "Hard". The classification is then performed using a K-Nearest Neighbors (KNN) classifier and further analyzed using several machine learning techniques and visualizations.

## Steps:

### 1. Data Preprocessing:
- Loaded the dataset from the file `Testing_carlease.csv`.
- Removed columns that had all missing values (`NA`).
- Saved the cleaned dataset as `data_cleaned.csv`.

### 2. Missing Data:
- Checked the cleaned dataset for any remaining missing values using `data_cleaned.isnull().sum()`.

### 3. Data Sorting:
- Sorted the data based on the "Time" column to categorize the entries into different clusters.

### 4. K-Means Clustering:
- Applied K-Means clustering with 3 clusters to classify the "Time" column into 3 categories: Easy, Medium, and Hard.
- Determined the cluster centers and used them to define thresholds for classification based on the "Time" values.

### 5. Label Encoding:
- Transformed the categorical "Difficulty" column into numerical labels (0, 1, 2) using `LabelEncoder` for machine learning compatibility.

### 6. Feature Selection:
- Selected numeric columns (excluding "Difficulty", "Time", and "cluster") as features for training the machine learning model.

### 7. Data Normalization:
- Applied Min-Max normalization to scale the feature values between 0 and 1 using `MinMaxScaler`.

### 8. Train-Test Split:
- Split the data into training (80%) and test (20%) sets using `train_test_split` for model evaluation.

### 9. K-Nearest Neighbors (KNN):
- Trained a KNN classifier with `n_neighbors=3` (selected based on experimentation) and evaluated it on the test data.
- Printed the accuracy score and classification report using `accuracy_score()` and `classification_report()`.

### 10. Visualization:
- Generated several visualizations to analyze the data:
  1. **Confusion Matrix**: Visualized the performance of the KNN model using a heatmap.
  2. **Boxplot**: Visualized the distribution of the "Time" variable to check for outliers.
  3. **Cluster Centers**: Visualized the K-Means cluster centers to display the boundaries for difficulty levels.
  4. **Feature Importance**: Visualized feature importance scores using a Random Forest model to assess which features most significantly affect classification.
  5. **ROC Curve** ROC CurveThe ROC curve for multi-class classification was drawn and the accuracy of the model for each class was evaluated.

### 11. Random Forest:
- Trained a Random Forest classifier to rank the importance of each feature in predicting the "Difficulty" label.
- Visualized the feature importance scores using a bar plot.

## Libraries Used:
- `numpy`
- `pandas`
- `sklearn` (for machine learning models and metrics)
- `matplotlib` (for plotting)
- `seaborn` (for advanced plotting)

## How to Run:
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
2. Place the Testing_carlease.csv file in the data directory.
3. Run the script to perform the analysis and visualizations.

## Results:
The script will output the accuracy of the KNN model, a classification report, and several plots visualizing the data and the results of the machine learning algorithms.

## Notes:
The code sorts the "Time" column before applying K-Means clustering to categorize the test tasks.
The dataset is preprocessed by removing columns with all missing values.
The KNN model is tuned with n_neighbors=3, based on experimentation.
Visualizations are generated for better understanding of the data, model performance, and feature importance.
Conclusion:
This project demonstrates a complete machine learning workflow from data preprocessing and clustering to classification and visualization. The results can be used to understand the different "Difficulty" levels based on the "Time" values and to gain insights from the dataset. The classification model can be further optimized and extended for more complex datasets.