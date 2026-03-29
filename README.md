
# ML Assignment - Supervised Learning

## Overview
In this repository, we will be comparing 4 different alogorithms against a common dataset and input variables in a regression problem of predicting annual medical insurance cost of a person.

## Contents
- Data preprocessing
- Model training and evaluation
- Performance metrics
- Results and analysis

## Getting Started
1. Clone this repository
2. Install required dependencies in each branch 
3. Run the scripts in the appropriate order in each branch

## Authors
Fernandopulle S.N.
Jameela M.J.F.
Luvinson I
Peiris M.S.M.

## Algortihms used:
1. Support Vector Regression (SVR)
2. Random Forest Regressor
3. Decision Tree Regressor
4. Gradient Boosting Regressor

## Dataset contains following features: 
person_id: Unique identifier for each individual
age: Age of the individual (18–90 years)
sex: Gender (Male/Female)
region: Geographic region
urban_rural: Whether the person lives in an urban or rural area
income: Annual income (USD)
education: Education attainment
marital_status: Marital status
employment_status: Employment type/status
household_size: Number of people in the household
dependents: Number of financial dependents
bmi: Body Mass Index value
smoker: Smoking status (Yes/No)
alcohol_freq: Frequency of alcohol consumption
visits_last_year: Number of healthcare provider visits in the last year
hospitalizations_last_3yrs: Number of hospitalizations in the past three years
days_hospitalized_last_3yrs: Total days spent in hospital over the past three years
medication_count: Number of active medications prescribed
systolic_bp: Systolic blood pressure (mmHg)
diastolic_bp: Diastolic blood pressure (mmHg)
ldl: LDL cholesterol level (mg/dL)
hba1c: Hemoglobin A1c percentage (indicator of blood sugar control)
plan_type: Type of insurance plan (e.g., HMO, PPO, EPO)
network_tier: Insurance network tier (Basic, Silver, Gold, Platinum)
deductible: Annual deductible amount in USD
copay: Copayment amount per healthcare service
policy_term_years: Length of the insurance policy in years
policy_changes_last_2yrs: Number of times the policy was modified in the past two years
provider_quality: Quality rating of the primary care provider/insurer
risk_score: Calculated health risk score (normalized)
annual_medical_cost: Total annual medical cost in USD
annual_premium: Annual insurance premium in USD
monthly_premium: Monthly insurance premium in USD
claims_count: Number of insurance claims filed in a year
avg_claim_amount: Average amount per claim in USD
total_claims_paid: Total value of claims paid in USD
chronic_count: Number of chronic conditions diagnosed
hypertension: Indicator for diagnosed hypertension (0 = No, 1 = Yes)
diabetes: Indicator for diabetes diagnosis (0 = No, 1 = Yes)
asthma: Indicator for asthma diagnosis (0 = No, 1 = Yes)
copd: Indicator for Chronic Obstructive Pulmonary Disease (0 = No, 1 = Yes)
cardiovascular_disease: Presence of cardiovascular disease (0 = No, 1 = Yes)
cancer_history: History of cancer diagnosis (0 = No, 1 = Yes)
kidney_disease: Indicator for chronic kidney disease (0 = No, 1 = Yes)
liver_disease: Indicator for liver disease (0 = No, 1 = Yes)
arthritis: Indicator for arthritis diagnosis (0 = No, 1 = Yes)
mental_health: Presence of a diagnosed mental health condition (0 = No, 1 = Yes)
proc_imaging_count: Number of imaging procedures (e.g., X-ray, MRI, CT scans) performed in a year
proc_surgery_count: Number of surgical procedures undergone
proc_physio_count: Number of psychiatric/psychological procedures or sessions
proc_consult_count: Number of specialist consultations (e.g., cardiologist, oncologist)
proc_lab_count: Number of laboratory diagnostic tests taken
is_high_risk: Binary indicator if the individual is classified as high health risk (0 = No, 1 = Yes)
