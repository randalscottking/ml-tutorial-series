# Data Directory

This directory contains the datasets used in the tutorial series.

## Datasets

### customer_churn.csv
Customer data for churn prediction (Tutorial 3)
- Rows: ~7,000 customers
- Features: 15 customer attributes
- Target: churned (1 = churned, 0 = active)

### employee_attrition.csv  
Employee data for attrition prediction (Tutorial 6)
- Rows: ~1,400 employees
- Features: 18 employee attributes
- Target: attrited (1 = left company, 0 = active)

## Data Generation

The datasets are synthetic but realistic, generated to match common business scenarios.

To generate the data files, run:

```bash
python scripts/generate_datasets.py
```

Alternatively, use the SQL scripts in the `sql/` directory to extract data from your own databases.

## Data Schema

### Customer Churn Data
- customer_id: Unique identifier
- tenure_months: Length of customer relationship
- monthly_charges: Monthly service fee
- total_charges: Total amount paid
- contract_type: Month-to-month, One year, Two year
- payment_method: Electronic check, Mailed check, etc.
- internet_service: DSL, Fiber optic, None
- tech_support: Yes, No
- online_security: Yes, No
- streaming_tv: Yes, No
- streaming_movies: Yes, No
- paperless_billing: Yes, No
- senior_citizen: 0 or 1
- partner: Yes, No
- dependents: Yes, No
- churned: 0 or 1 (target)

### Employee Attrition Data
- employee_id: Unique identifier
- age: Employee age
- years_at_company: Tenure in years
- years_in_role: Time in current role
- years_with_manager: Time with current manager
- monthly_income: Monthly salary
- job_satisfaction_rating: 1-5 scale
- environment_satisfaction_rating: 1-5 scale
- work_life_balance_rating: 1-5 scale
- performance_rating: 1-5 scale
- department: Sales, R&D, HR
- job_role: Manager, Engineer, etc.
- education_level: 1-5 (high school to doctorate)
- marital_status: Single, Married, Divorced
- distance_from_home_km: Commute distance
- overtime_hours_per_month: Average overtime
- training_times_last_year: Number of training sessions
- promotions_last_5_years: Number of promotions
- attrited: 0 or 1 (target)

## Notes

- Missing values are intentionally included to teach handling techniques
- Class imbalance reflects real-world scenarios
- Features selected for educational value and business relevance
