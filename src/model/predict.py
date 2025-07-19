import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/salary_scaler.pkl')

# Defines the exact features used during training
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

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')
X_test = X_test[features_used]  # Ensure columns are aligned

# Make scaled predictions
pred_scaled = model.predict(X_test)

# Prepare input for inverse_transform
dummy_input = pd.DataFrame({
    'years_experience': [0] * len(pred_scaled),
    'salary_usd': pred_scaled
})

# Inverse transform to get actual salary predictions
inv_transformed = scaler.inverse_transform(dummy_input)
actual_salaries = inv_transformed[:, 1]  # Extract only salary column

# Save final predictions
pd.DataFrame(actual_salaries, columns=['Predicted_Salary']).to_csv('data/predictions/predictions.csv', index=False)
print("Predictions saved to predictions.csv")
