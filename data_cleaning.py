# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:12:11 2021

@author: Sheryar Kahout
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

#salary parsing
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1'] # -1 in '' because Salary estimate is an object or a text field and not a numeric.

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

#hourly wage to annual wage
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly == 1 else x.min_salary, axis =1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly == 1 else x.max_salary, axis =1)
df[df.hourly == 1][['hourly','min_salary','max_salary']]

#Company name text only
df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)
df['company_text'] = df['company_text'].apply(lambda x: x.replace('\r\n', ''))
df.company_text

#State field
df['job_state']= df['Location'].apply(lambda x: x.split(',')[1])

#fixing state Los Angeles
df['job_state'] = df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df.job_state.value_counts()

# NOTE
# df.job_state vs the df['job_state'] is the same thing but if there is a space in job_state, we can't do the dot method.
df['same_state'] = df.apply(lambda x: 1 if x.Headquarters == x.Location else 0, axis = 1) 

#Age of Company
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2021 - x)

#Parsing of job description (python, RStudio, Exel)
df['Job Description'][0]
#length of job secription
df['des_len'] = df['Job Description'].apply(lambda x: len(x))
df['des_len']

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python_yn.value_counts()

#r studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r language' in x.lower() else 0)
df.R_yn.value_counts()

#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark_yn.value_counts()

#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws_yn.value_counts()

#exel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel_yn.value_counts()

#tableau
df['tableau_yn'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df.tableau_yn.value_counts()

#Competitor's Count
df['comp_count'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)
df['comp_count']

df.to_csv('salary_data_cleaned.csv', index = False)


    












