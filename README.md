## Overview

JobScope-AI is an end-to-end project that combines **data analysis**, **machine learning**, and **web development** to predict estimated salaries for AI-related job roles. The app takes in job details such as title, company size, education level, and experience, and returns a salary prediction using trained machine learning models.

---

## 📊 Machine Learning Models Used

Three different regression models were trained and evaluated:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Each model was assessed using standard regression metrics:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

📈 **Gradient Boosting Regressor** consistently outperformed the others across all evaluation metrics and was selected as the final model for deployment.

---

## 🧰 Tech Stack

### 🧪 Data Science
- **Pandas** – data manipulation and preprocessing
- **Scikit-learn** – model training, evaluation, and preprocessing
- **Jupyter Notebook** – exploratory analysis and model experimentation

### 🌐 Web Development
- **Flask** – Python web framework for serving the model
- **HTML/CSS** – frontend form and layout

