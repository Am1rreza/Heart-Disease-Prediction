# Heart Disease Prediction using XGBoost

## Project Overview
This project aims to predict the 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset. It demonstrates data preprocessing, balancing techniques, and model training with XGBoost.

## Dataset
The dataset includes features such as age, sex, smoking habits, blood pressure, cholesterol levels, BMI, glucose, and others, along with the target label `TenYearCHD` indicating the presence of heart disease within 10 years.

## Key Steps in the Project

1. **Data Cleaning & Preparation:**  
   - Remove missing values  
   - Drop irrelevant features (e.g., 'education')  
   - Rename columns for clarity  
   - Scale features using `StandardScaler`

2. **Train-Test Split:**  
   - Use stratified split to keep label distribution balanced

3. **Handling Imbalanced Data:**  
   - Apply SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class in the training set

4. **Modeling with XGBoost:**  
   - Use XGBoost classifier with tuned parameters  
   - Calculate and set `scale_pos_weight` to balance class weights during training

5. **Evaluation:**  
   - Calculate accuracy, precision, recall, and F1-score  
   - Display confusion matrix heatmap using Seaborn

## Libraries Used
- pandas, numpy  
- scikit-learn  
- imbalanced-learn  
- xgboost  
- matplotlib, seaborn

## Notes
- The dataset is imbalanced, with fewer positive cases of heart disease. Handling this imbalance is critical for improving model sensitivity to the minority class.  
- `scale_pos_weight` helps XGBoost focus more on correctly classifying the minority class.
