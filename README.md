https://github.com/user-attachments/assets/e8b76f3d-dabd-4268-9ae2-d47e50541c96


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

---

## Running Web App
1. Clone the repo
2. cd jobscope-ai
3. setup virtual enviornment:
   - python3 -m venv venv
   - source venv/bin/activate   # On Windows use: venv\Scripts\activate
5. pip install -r requirements.txt
6. in terminal : python3 app/api.py
7. (If python3 does not work do python, vice versa)
