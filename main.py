# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:59:44 2021

@author: Sheryar Kahout
"""

import glassdoor_scraper as gs
import pandas as pd

path =  "C:/Users/Sheryar Kahout/Documents/ds_salary_proj/chromedriver"

df = gs.get_jobs('data scientist', 15, False, path, 15)
