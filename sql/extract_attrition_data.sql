-- Extract employee attrition data from HR database
-- This query pulls employee features and attrition status

SELECT 
    employee_id,
    age,
    years_at_company,
    years_in_role,
    years_with_manager,
    monthly_income,
    job_satisfaction_rating,
    environment_satisfaction_rating,
    work_life_balance_rating,
    performance_rating,
    department,
    job_role,
    education_level,
    marital_status,
    distance_from_home_km,
    overtime_hours_per_month,
    training_times_last_year,
    promotions_last_5_years,
    attrited  -- Target variable: 1 if left company, 0 if active
FROM 
    employees
WHERE 
    -- Filter for employees with minimum tenure
    years_at_company >= 1
    AND monthly_income IS NOT NULL
ORDER BY 
    employee_id;

-- Expected output: ~1400 rows with 19 columns
-- Use this data for Tutorial 6: Employee Attrition Prediction
