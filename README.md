# Salary Estimator for Data Science Jobs in America<br />
* Created a tool that estimates data science salaries (MAE ~ $ 12K) to give a general overview of what to expect coming into data science and for new hires to negotiate their income.<br />
* Cleaned data worth approximately 1000 entries extracted from glassdoor regarding data science jobs.<br />
* Engineered features from the text of each job description to quantify the value companies put on Python, R, Spark, AWS, Excel, Tableau.<br />
* Used GridsearchCV to optimize MLR, Lasso and Random Forest Regressor to reach the best model<br />

## Resources<br />
**Python Version:** 3.7<br />
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn<br />

## Data Set Features Initial<br />
* Job title<br />
* Salary Estimate<br />
* Job Description<br />
* Rating<br />
* Company Name<br />
* Location<br />
* Company Headquarters<br />
* Size of Company<br />
* Founded Date<br />
* Type of Ownership<br />
* Industry<br />
* Sector<br />
* Revenue<br />
* Competitors<br />

## Data Cleaning<br />
Cleaned the data to prepare for EDA by making the following changes and variable creation:<br />

* Parsed Salary for numeric data<br />
* Differentiated between hourly wages and employer provided salary<br />
* Omitted missing salary data<br />
* Parsed company text to extract Ratings<br />
* Calculated age of company from the Founded column<br />
* Length of job description<br />
* Parsed company description to extract and categorize data based on the following skill sets:<br />
* Python<br />
* R<br />
* Spark<br />
* AWS<br />
* Excel<br />
* Tableau<br />

## EDA<br />
* Profiled data using Pandas Profiling and visualized for a clear outline of data distribution. (filename : output.html)
* Added sections for job seniority to better differentiate between characteristics
* Following are some of the highlights of the process

## Model Building
* lasso: Because there is a normalization effect and we have a sparse matrix
* Rf: Because we have a lot of 0 and 1 values and that is a good use case because we are using a bunch of decision trees

