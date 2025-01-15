# DECISION-TREE-IMPLEMENTATION_CODTECH
**COMPANY** : COSTECH IT SOLUTIONS
**NAME** : DHANUSHNI.N
**INTERN ID** : CT08FCX
**DOMAIN** : MACHINE LEARNING
**BATCH DURATION** :DECEMBER 20th ,2024 TO JANUARY 20th,2025
**MENTOR NAME** : NEELA SANTHOSH KUMAR

#Decision Tree Model for Titanic Dataset
This project focuses on building and visualizing a Decision Tree Classifier to analyze and predict the survival of passengers on the Titanic using the Scikit-learn library in Python. The dataset used is a well-known Titanic dataset, which contains information about passengers such as age, gender, class, and survival status. This project demonstrates the process of data preprocessing, model training, visualization, and evaluation.

**Objective**
The primary objective of this project is to classify passengers as either "Survived" or "Not Survived" based on their features. The task involves creating a decision tree model to achieve this classification and visualizing the tree to understand the decision-making process.

**Steps Performed**
1. Data Loading
The Titanic dataset was loaded from a public online source, specifically a CSV file hosted on GitHub. The dataset consists of multiple features, including passenger demographics, ticket class, family relationships, and survival status.

2. Data Preprocessing
Data preprocessing was a critical step to ensure the dataset was clean and suitable for model training. The following preprocessing tasks were performed:

**Handling Missing Values:**
Missing values in the Age column were replaced with the median age.
Missing values in the Embarked column were replaced with the mode, as it represents the most frequent boarding location.
Feature Selection:
Selected key features for training the model: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.
The target variable (Survived) was kept separate.
Encoding Categorical Variables:
The Sex column was encoded numerically (male = 0, female = 1).
The Embarked column was encoded numerically (C = 0, Q = 1, S = 2).
3. Splitting the Dataset
The cleaned dataset was split into training and testing sets. A 70-30 split was used to train the model on a significant portion of the data while leaving enough unseen data for evaluation. The train_test_split function from Scikit-learn was utilized for this purpose.

4. Model Training
A decision tree classifier was implemented using Scikit-learn’s DecisionTreeClassifier. The model was configured with the following parameters:

Criterion: Gini impurity, which measures the quality of splits.
Max Depth: 4, to prevent overfitting and maintain model interpretability.
The model was trained on the training dataset to learn patterns and relationships between the features and the survival outcome.

5. Model Visualization
The trained decision tree was visualized using Scikit-learn’s plot_tree function. The visualization provides a graphical representation of the tree structure, showing how features and thresholds contribute to classification decisions. The visualization included:

Node details: Feature names, thresholds, Gini impurity, and class distribution.
Color-coding for easy interpretation.
6. Model Evaluation
The model's performance was evaluated on the test set using the following metrics:

Accuracy Score: To measure the overall correctness of predictions.
Classification Report: To provide detailed metrics such as precision, recall, and F1-score for each class.
Confusion Matrix: To display the count of true positives, true negatives, false positives, and false negatives.
A heatmap was created using Seaborn to visualize the confusion matrix, offering insights into the model's prediction behavior.

7. Insights and Conclusion
The decision tree model provided an interpretable approach to classify Titanic passengers. Visualization of the decision tree helped understand how the model prioritized features like gender and ticket class. The evaluation metrics demonstrated the model's effectiveness, while the confusion matrix revealed areas for improvement.

Technologies Used
Programming Language: Python
Libraries:
Pandas and NumPy for data manipulation.
Scikit-learn for machine learning and model building.
Matplotlib and Seaborn for data visualization.
How to Use
Clone this repository to your local system.
Ensure all dependencies listed in requirements.txt are installed.
Run the Jupyter Notebook or Python script to view the results.

