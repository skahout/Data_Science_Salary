# Salary Estimator for Data Science Jobs<br />
* Created a tool that estimates data science salaries (MAE ~ $ 9.9K) to give a general overview of what to expect coming into data science and for new hires to negotiate their income.<br />
* Cleaned data worth approximately 1000 entries extracted from glassdoor regarding data science jobs.<br />
* Engineered features from the text of each job description to quantify the value companies put on Python, R, Spark, AWS, Excel, Tableau.<br />
* Used GridsearchCV to optimize Decision Tree and Random Forest Regressor to reach the best model<br />

## Resources<br />
**Python Version:** 3.7<br />
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn<br />

## Data Set Features (Initial)<br />
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

![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/avg_salary_hist.png "Average Salary Histogram")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/factor_heatmap.png "Factor Correlation Heatmap")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/factor_variations_boxplot.png "Factor Variation Boxplot")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/rating_boxplot.png "Ratings Boxplot")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/job_type_hist.png "Histogram of Job Type")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/job_state_hist.png "Histogram of Job State")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/ownership_hist.png "Histogram of Ownership")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/section_hist.png "Histogram of Section")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/worcloud.png "Data Science Word Cloud")
![alt text](https://github.com/skahout/ds_salary_proj/blob/main/png/pearson_r_correlation.PNG "Pearson's r for Factors")

## Model Building
* Lasso: Because there is a normalization effect and we have a sparse matrix
* Random Forest: Because we have a lot of 0 and 1 values and that is a good use case because we are using a bunch of decision trees
* In Addition performed Linear Regression, Decision Tree, Support Vector Regression

## Model Performance
Both the Decision Tree and Random Forest gave the best and approximately same results on the test set, however Decision Tree outperformed the other models during validaiton. Following are the Mean Absolute Error values by their respective models.
* Linear Regression : 18.8k
* Random Forest : 12k
* Decision Tree : 9.9k
* Lasso Regression : 19k
* Support Vector Regression : 29k
