# Salary Estimator for Data Science Jobs in America
*Created a tool that estimates data science salaries (MAE ~ $ 12K) to give a general overview of what to expect coming into data science and new hires to negotiate their income.
*Cleaned data worth approximately 1000 entries extracted from glassdoor regarding data science jobs.
*Engineered features from the text of each job description to quantify the value companies put on Python, R, Spark, AWS, Excel, Tableau.
*Used GridsearchCV to optimize MLR, Lasso and Random Forest Regressor to reach the best model

#Resources
**Python Version:** 3.7
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

#Data Set Features Initial
*Job title
*Salary Estimate
*Job Description
*Rating
*Company Name
*Location
*Company Headquarters
*Size of Company
*Founded Date
*Type of Ownership
*Industry
*Sector
*Revenue
*Competitors

#Data Cleaning
Cleaned the data to prepare for EDA by making the following changes and variable creation:

*Parsed Salary for numeric data
*Differentiated between hourly wages and employer provided salary
*Omitted missing salary data
*Parsed company text to extract Ratings
*Calculated age of company from the Founded column
*Length of job description
*Parsed company description to extract and categorize data based on the following skill sets:
*Python
*R
*Spark
*AWS
*Excel
*Tableau
