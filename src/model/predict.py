import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/salary_scaler.pkl')

# Define features used in training (used to align test set)
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

# Load processed X_test for predictions
X_test = pd.read_csv('data/processed/X_test.csv')
X_test_model_input = X_test[features_used]

# Predict scaled salaries
pred_scaled = model.predict(X_test_model_input)

# Prepare for inverse scaling (salary was scaled with years_experience as a pair)
dummy_input = pd.DataFrame({
    'years_experience': [0] * len(pred_scaled),
    'salary_usd': pred_scaled
})
inv_transformed = scaler.inverse_transform(dummy_input)
actual_salaries = inv_transformed[:, 1]  # only salary column

X_test_human = pd.read_csv('data/processed/X_test_readable.csv')  # This will have 6000 rows
X_test_human['Predicted_Salary'] = actual_salaries  # Lengths now match
X_test_human.to_csv('data/predictions/final_predictions.csv', index=False)
print("Final predictions saved with human-readable labels.")
