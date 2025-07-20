from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/gradient_boosting_model.pkl")
scaler = joblib.load("models/salary_scaler.pkl")

# Features that were used
features_used = [
    'experience_level', 'company_size', 'education_required', 'years_experience',
    'industry_Automotive', 'industry_Consulting', 'industry_Education', 'industry_Energy',
    'industry_Finance', 'industry_Gaming', 'industry_Government', 'industry_Healthcare',
    'industry_Manufacturing', 'industry_Media', 'industry_Real Estate', 'industry_Retail',
    'industry_Technology', 'industry_Telecommunications', 'industry_Transportation',
    'title_AI Architect', 'title_AI Consultant', 'title_AI Product Manager',
    'title_AI Research Scientist', 'title_AI Software Engineer', 'title_AI Specialist',
    'title_Autonomous Systems Engineer', 'title_Computer Vision Engineer', 'title_Data Analyst',
    'title_Data Engineer', 'title_Data Scientist', 'title_Deep Learning Engineer', 'title_Head of AI',
    'title_ML Ops Engineer', 'title_Machine Learning Engineer', 'title_Machine Learning Researcher',
    'title_NLP Engineer', 'title_Principal Data Scientist', 'title_Research Scientist', 'title_Robotics Engineer'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract values from form
    exp_level = int(request.form["experience_level"])
    company_size = int(request.form["company_size"])
    edu_level = int(request.form["education_required"])
    yoe = float(request.form["years_experience"])
    job_title = request.form["job_title"]
    industry = request.form["industry"]

    # Start building input row
    input_dict = {col: 0 for col in features_used}
    input_dict["experience_level"] = exp_level
    input_dict["company_size"] = company_size
    input_dict["education_required"] = edu_level
    input_dict["years_experience"] = yoe
    input_dict[job_title] = 1
    input_dict[industry] = 1

    input_df = pd.DataFrame([input_dict])

    # Predict on scaled data
    pred_scaled = model.predict(input_df)

    # Inverse transform to get actual salary
    dummy_input = pd.DataFrame({
        "years_experience": [0],
        "salary_usd": pred_scaled
    })
    actual_salary = scaler.inverse_transform(dummy_input)[0, 1]
    predicted_salary = round(actual_salary, 2)

    return render_template("index.html", prediction=predicted_salary)

if __name__ == "__main__":
    app.run(debug=True)
