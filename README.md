
# Predicting Employee Attrition Using Workforce Analytics (IBM HR Analytics Dataset)

This project aims to predict employee attrition using various workforce analytics, such as workplace conditions, demographic information, and employee performance factors. The goal is to compare the performance of different machine learning models, including ensemble methods and linear models, to predict the likelihood of an employee leaving the company. The project also investigates the most significant factors affecting employee attrition using SHAP analysis for result interpretation. Issues like fairness and class imbalance in predictive modeling are addressed.

## **Research Questions**
1. Can machine learning models accurately predict employee attrition, and how do ensemble methods compare to linear models?
2. Which workplace and demographic factors most significantly influence employee attrition, and can SHAP analysis help explain these factors?
3. How can predictive models balance organizational decision-making with fairness and bias considerations, especially when addressing class imbalance?

## **Dataset Details**
- **Name:** IBM HR Analytics Employee Attrition & Performance Dataset
- **Source:** [Kaggle – IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Contributors:** This dataset is collected from IBM's internal HR surveys and made publicly available on Kaggle. It contains data on employees' demographics, performance, satisfaction levels, and reasons for leaving the company.

## **Libraries Used**
- **pandas** for data manipulation and analysis.
- **matplotlib** and **seaborn** for data visualization.
- **scikit-learn** for machine learning models and evaluation metrics.
- **SHAP** for model interpretability and feature importance analysis.

## **Steps Performed in the Notebook**
1. **Import Libraries:** All necessary libraries for data manipulation, machine learning, and evaluation are imported.
2. **Load Dataset:** The dataset is loaded and previewed to ensure proper formatting and data types.
3. **Exploratory Data Analysis (EDA):** Basic statistical analysis and visualizations are performed to understand the distribution of data and relationships between features.
4. **Data Preprocessing:**
   - Missing values are handled.
   - Categorical variables are encoded, and features are scaled for model training.
5. **Model Training:**
   - Multiple machine learning models are trained, including **Logistic Regression**, **SVM**, **KNN**, **Random Forest**, and **Gradient Boosting**.
   - A **Voting Classifier** and **Stacking Classifier** are also implemented to improve performance.
6. **Hyperparameter Tuning:** The models are tuned using techniques like **GridSearchCV** and **RandomizedSearchCV** to optimize parameters for better performance.
7. **Model Evaluation:**
   - Various metrics like **accuracy**, **precision**, **recall**, **F1-score**, **AUC-ROC**, and **confusion matrix** are computed to evaluate the models.
8. **SHAP Analysis:** SHAP values are used to interpret the model and identify which features influence the prediction of employee attrition.
9. **Fairness and Bias Consideration:** The class imbalance problem is addressed using techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) and **balanced class weights**.

## **How to Run the Notebook**
1. Clone the repository to your local machine.
2. Install the required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn shap
   ```
3. Download the dataset from Kaggle or use your own structured dataset similar to the IBM HR Analytics dataset.
4. Open the notebook in **Jupyter Notebook** or **Google Colab**.
5. Execute all the cells to perform data analysis, model training, and performance comparison.

## **Results**
The project compares multiple machine learning models for predicting employee attrition and evaluates them based on various performance metrics. SHAP analysis provides insights into the most influential factors affecting employee attrition.

## **Contributions**
Feel free to contribute by:
- Adding new machine learning algorithms.
- Enhancing data preprocessing techniques.
- Improving the handling of class imbalance and fairness considerations.

## **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
