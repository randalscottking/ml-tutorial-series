-- Extract customer churn data from database
-- This query pulls customer features and churn status

SELECT 
    customer_id,
    tenure_months,
    monthly_charges,
    total_charges,
    contract_type,
    payment_method,
    internet_service,
    tech_support,
    online_security,
    streaming_tv,
    streaming_movies,
    paperless_billing,
    senior_citizen,
    partner,
    dependents,
    churned  -- Target variable: 1 if churned, 0 if active
FROM 
    customers
WHERE 
    -- Filter out incomplete records
    total_charges IS NOT NULL
    AND tenure_months > 0
ORDER BY 
    customer_id;

-- Expected output: ~7000 rows with 16 columns
-- Use this data for Tutorial 3: Customer Churn Prediction
