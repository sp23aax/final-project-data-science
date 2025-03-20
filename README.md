# Nutritional Data Analysis and Prediction Project

## 1. Project Title
**Nutritional Data Analysis and Prediction Using Machine Learning**

## 2. Project Description
This project focuses on analyzing nutritional data from multiple datasets and predicting caloric values using various machine learning models. The primary goal is to build a robust predictive model that can accurately estimate the caloric value of food items based on their nutritional content (Fat, Protein, and Carbohydrates). The project implements data cleaning, visualization, feature engineering, dimensionality reduction, and multiple regression models.

### Technologies Used:
- **Pandas** and **NumPy** for data manipulation and numerical operations.
- **Matplotlib** and **Seaborn** for data visualization.
- **Scikit-learn** for machine learning models, data preprocessing, and evaluation.
- **Joblib** for saving and loading models.

### Challenges Faced:
- Handling missing and non-numeric values in the dataset.
- Dealing with multicollinearity among features.
- Identifying the most effective regression model for predicting caloric values.
- Hyperparameter tuning to optimize model performance.

### Future Enhancements:
- Add more food-related features to improve prediction accuracy.
- Integrate a web-based interface for real-time predictions.
- Explore deep learning techniques for enhanced model performance.

## 3. Table of Contents:3. Table of Contents

1.Project Title

2.Project Description

3.Table of Contents

4.How to Install and Run the Project

5.Data Preprocessing

6.Exploratory Data Analysis (EDA)

7.Model Training and Evaluation

8.Hyperparameter Tuning

9.Feature Importance

10.Model Saving and Loading

## 4. How to Install and Run the Project
### Prerequisites:
Ensure you have Python 3.x installed along with the following libraries:
                                                                        - pandas
                                                                        - numpy
                                                                        - matplotlib
                                                                        - seaborn
                                                                        - scikit-learn
                                                                        - joblib

### Installation:
1. Clone the repository:https://github.com/sp23aax/final-project-data-science
2. Navigate to the project directory:https://github.com/sp23aax/final-project-data-science/blob/main/final_code.ipynb
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project:
1. Load the datasets into the project folder.
2. Execute the main Python script:
   ```bash
   python main.py
   ```
3. The results, including model performance metrics and visualizations, will be displayed in the terminal or as plots.

## 5. Data Preprocessing
- Importing necessary libraries.
- Loading CSV files and concatenating datasets.
- Checking for missing values and handling them by filling with mean values.
- Dropping duplicate rows.

## 6. Exploratory Data Analysis (EDA)
- Descriptive statistics.
- Visualizing feature distributions using histograms.
- Identifying outliers using boxplots.
- Correlation analysis using a heatmap.

## 7. Model Training and Evaluation
### Models Implemented:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Support Vector Machine (SVM)**

### Evaluation Metrics:
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## 8. Hyperparameter Tuning
Hyperparameter tuning was performed using **GridSearchCV** to optimize the performance of the Random Forest model. The best hyperparameters were identified and used for the final model.

## 9. Feature Importance
The feature importance for the Random Forest model was visualized to understand the contribution of each nutritional feature (Fat, Protein, Carbohydrates) to the caloric value prediction.

## 10. Model Saving and Loading
- The trained Random Forest model and the scaler were saved using **Joblib**.
- The saved model can be loaded later for making predictions on new data without retraining.

## Learning Curve and Residual Analysis
The project also includes a learning curve for the Random Forest model and a residual plot to analyze prediction errors.

