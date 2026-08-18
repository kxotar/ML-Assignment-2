# Dry Bean Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and compare multiple machine-learning classification models for predicting the variety of a dry bean from its physical and morphological measurements.

Five classification algorithms were trained and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

The models were compared using accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient. An interactive Streamlit application was also developed to allow users to upload test data, select a model, view predictions, and compare model performance.

## b. Dataset Description

### Dataset name

Dry Bean Dataset

### Dataset source

UCI Machine Learning Repository:  
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

### Dataset overview

The Dry Bean Dataset contains measurements obtained from images of seven registered varieties of dry beans. A computer-vision system was used to extract numerical features describing each bean's dimensions and shape.

| Property | Description |
|---|---:|
| Number of instances | 13,611 |
| Number of input features | 16 |
| Target variable | `Class` |
| Number of target classes | 7 |
| Task | Multiclass classification |
| Feature types | Integer and continuous numerical features |
| Missing-value handling | Median imputation within each model pipeline |

### Target classes

The target variable contains the following seven dry bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Input features

The dataset contains the following 16 input features:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRatio
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- Roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

The dataset meets the assignment requirement of at least 500 instances and at least 12 features.

## c. Project Links

- **GitHub repository:** https://github.com/kxotar/ML-Assignment-2
- **Live Streamlit application:** https://ml-assignment-2-muawvzxu86ukgwwfuuoaxb.streamlit.app/
- **Dataset source:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## d. Models Used

- **Logistic Regression :**
Logistic Regression is a linear classification algorithm that estimates the probability of each target class. Standardized input features were used, and the maximum number of optimization iterations was increased to support convergence.

- **Decision Tree Classifier :**
A Decision Tree predicts the target class by repeatedly dividing the feature space according to decision rules. It can learn nonlinear relationships and is easy to interpret, but an unrestricted tree may overfit the training data.

- **K-Nearest Neighbors Classifier :**
K-Nearest Neighbors classifies an observation using the classes of its nearest observations in the feature space. The model used `k = 5`. Feature standardization was applied because distance-based algorithms are sensitive to differences in feature scale.

- **Gaussian Naive Bayes :**
Gaussian Naive Bayes uses Bayes' theorem and assumes that each numerical feature follows a Gaussian distribution within each class. It is computationally efficient, although its feature-independence assumption may not fully represent relationships among bean measurements.

- **Random Forest Classifier :**
Random Forest is an ensemble model consisting of multiple decision trees. The final prediction is determined by combining the predictions from the individual trees. The model used 300 trees and a fixed random state of `42`.

### Evaluation Methodology

The five models were evaluated on the same stratified test set using the following metrics:

- **Accuracy:** Proportion of test records classified correctly.
- **AUC Score:** Ability of the model to rank and distinguish the target classes.
- **Precision:** Proportion of predicted class assignments that were correct.
- **Recall:** Proportion of actual class records correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC Score:** Correlation between actual and predicted classes, considering all parts of the confusion matrix.

### Model Comparison

The following results were obtained on the same stratified test dataset. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | **0.9948** | **0.9215** | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| K-Nearest Neighbors | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Random Forest | **0.9210** | 0.9934 | 0.9211 | **0.9210** | **0.9210** | **0.9045** |


## 8. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression produced highly competitive results, with an accuracy of 0.9207 and an F1 score of 0.9209. It achieved the highest AUC score of 0.9948 and the highest weighted precision of 0.9215. This indicates that the standardized features allowed the linear model to distinguish the bean classes effectively. Its performance was only marginally below Random Forest on accuracy, recall, F1, and MCC. |
| Decision Tree | Decision Tree achieved an accuracy of 0.8920, an F1 score of 0.8916, and an MCC score of 0.8696. These were the lowest values among the five evaluated models. Its AUC of 0.9450 was also the lowest. A single unrestricted tree may have learned rules that did not generalize as effectively to the test data. |
| K-Nearest Neighbors | K-Nearest Neighbors achieved an accuracy of 0.9166, an F1 score of 0.9168, and an MCC score of 0.8992. It performed better than Gaussian Naive Bayes and Decision Tree but remained slightly behind Logistic Regression and Random Forest. Feature standardization was important because KNN predictions depend on distances between observations. |
| Gaussian Naive Bayes | Gaussian Naive Bayes achieved an accuracy of 0.8979 and an F1 score of 0.8981. Its AUC score was notably high at 0.9916, despite its lower accuracy and F1 score. This suggests that the model ranked class probabilities effectively, but its final class assignments were less accurate than those of Random Forest, Logistic Regression, and KNN. Correlations among the morphological features may have limited the effectiveness of its feature-independence assumption. |
| Random Forest | Random Forest achieved the highest accuracy of 0.9210, highest recall of 0.9210, highest F1 score of 0.9210, and highest MCC score of 0.9045. Its AUC score of 0.9934 was also very strong. Combining predictions from multiple decision trees allowed it to generalize substantially better than the individual Decision Tree. |
| Overall Winner | Random Forest was selected as the overall winner because it obtained the highest accuracy, recall, weighted F1 score, and MCC score. Logistic Regression remained a very close competitor and achieved the highest AUC and precision. |

### Conclusion

Random Forest was selected as the best overall model for the Dry Bean Dataset. It achieved an accuracy of **0.9210**, an AUC score of **0.9934**, a weighted precision of **0.9211**, a weighted recall of **0.9210**, a weighted F1 score of **0.9210**, and an MCC score of **0.9045**.

Random Forest produced the highest result in four of the six required evaluation metrics: accuracy, recall, F1, and MCC. Logistic Regression achieved the highest AUC and precision and performed almost as well overall. The small difference between these two models shows that both ensemble and linear classification approaches were effective for this dataset. Based on the combined evaluation results, Random Forest provided the strongest overall performance.
