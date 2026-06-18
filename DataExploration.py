import pandas as pd
import numpy as np


df = pd.read_csv('Loan_Data.csv') # data injection - csv file is loaded
print(df)

# ==========================================================================================================================================================
# #1. Data understanding. this includes checking the dataset column names, the dataset nomenclature and business problem
# ==========================================================================================================================================================
# ==========================================================================================================================================================
#2. Data Exploration
# ==========================================================================================================================================================
# ==========================================================================================================================================================


print(f"🟢Shape is : {df.shape}")    # to get the dimensions. Will output the total number of columns & rows.    
print(f"🟢Column names are listed as : \n {df.columns.to_list()}") #to list the column names in the dataframe.

# =============================================================================
# if the column name syntax is different from variable name syntax, we should change it using rename. Notice the all lower case and irregular format of the col names
# df_renamed = df.rename(columns = {'col_Name':'New_col_name','second_col_name':'New_Second_col_name'})
# =============================================================================

unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
print(f"🟢Unique Columns in the df are : {unique_cols}")

# =============================================================================
# print(df.columns.nunique()) #the number of unique column names
# print(df.nunique()) #number of unique values in all the columns 
# these uniques columns should be deleted as they have no impact on the predictive analysis since they have no pattern but delete it post the data cleaning
# this is called dropping unimportant columns
# =============================================================================

print(df.info())
print(f"🟢This is the use of head :\n {df.head()}")
print(f"🟢This is the use of head(5) to show the first 5 rows - \n {df.head(5)}")
print(f"🟢This is for displaying null values in the columns \n {df.isnull().sum()}")
print(f"🟢This is for displaying null values percentage in the columns \n {df.isnull().sum()/len(df)*100}")

print(f"🟢The duplicate column names are :\n {df.duplicated().sum()}")

# this is for listing the unique values and their datatypes to find the type of values within a column
for col in df.columns:
    unique_values = df[col].unique()
    count = len(unique_cols)
    print(f"🟢{unique_values} | {count}")
    
# to combine the primary and secondary income sources - ApplicantIncome	+CoapplicantIncome

df['Income'] =  df['ApplicantIncome']+df['CoapplicantIncome']
df.drop(columns = ['ApplicantIncome','CoapplicantIncome'], inplace = True)

print(df)

#to find the numercial and categorical columns
print(f"🟢the list of numerical columns are : \n {df.select_dtypes(include = ['int64','float64']).columns.to_list()}")
print(f"🟢the list of categorical columns are : \n {df.select_dtypes(include=['str', 'category']).columns.tolist()}")

# =============================================================================
# # to separate the numerical columns into continuous and count
# continuous_cols = df.select_dtypes(include=['float64']).columns.tolist()
# 
# # Count columns are strictly integers (whole numbers)
# count_cols = df.select_dtypes(include=['int64']).columns.tolist()
# 
# print("Continuous (Floats):", continuous_cols)
# print("Count (Integers):", count_cols)

# THIS WONT WORK BECAUSE OF ANAMOLIES IN THE DATE. SO TO CORRECT THAT WE NEED TO CLEAN THE DATA IN THE INDIVIDUAL COLUMNS
# =============================================================================

df['Dependents'] = df['Dependents'].replace({'3+':'3'})
df['Dependents'] = pd.to_numeric(df['Dependents'],errors='coerce')

numerical_cols = df.select_dtypes(include = ['int64','float64'])

continous = []
count = []

threshold = 15

for col in numerical_cols:
    uniques = df[col].dropna().nunique()
    if uniques <= threshold:
        count.append(col)
    else:
        continous.append(col)
        
print("🟢 Continuous Columns (Infinite/Smooth ranges):")
print(continous)

print("\n🔵 Count / Discrete Columns (Fixed whole increments):")
print(count)

print("\n🟡 Handled Separately (Binary/Categorical Flags):")
print(['Credit_History'])        

#to describe the column's count, mean, sd, min, percentiles(Q1,Q2 called median, Q3) and max
# =============================================================================
# 
# print(df[continous].describe())
# print(df[count].describe())
# =============================================================================
print(df[continous + count].describe())
categorical = df.select_dtypes(include=['str', 'category']).columns.tolist()
print(df[categorical].describe())

#this will provide the describe function to the entire df but only numerical columns will be applied the formula
print(df.describe()) 


