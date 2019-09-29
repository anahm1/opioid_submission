#!/usr/bin/env python
# coding: utf-8

# In[1]:


#I used the following general methodology for analysis / prediction:

    #Imported CSV and converted to data frame
    #Performed exploratory data analysis
    #Cleansed data
    #Random Forest Classifier algorithm
 
# The model selects a sample of claimants from the original dataset and predicts opiod use each time the script is run
# The script generates the output in a .csv file "opiodusepredictionresults.csv"
#I have used a random forrest classifier model to predict opiod use by claimants






# In[2]:


#import libraries

#Multidimensional Array and Matrix Representation Library needed to input cleansed data into model
import numpy as np 

# Python Data Analysis Library for Data Frame, CSV File I/O 
import pandas as pd 

#Data Visualization Library
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Data Visualization Library built on top of Matplotlib
#a cleaner visualization and easier interface to call
import seaborn as sns


#Algorithms and accuracy testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz


# In[3]:


#DATA IMPORT, INSPECTION AND CLEANSING


#i'm going to standardize any missing values in 
#our data frame so that all missing values are picked up i.e. n/a, na, Naan 
missing_values = ["n/a", "na", "--"]

#import file and define as maindf, i've renamed the original file to 'train.csv' 
#for ease of use. This will be the file that will train the model
maindf = pd.read_csv('train.csv',low_memory=False, na_values = missing_values)


# In[4]:


#inspect the data by looking at the top 5 rows
maindf.head()


# In[5]:


#inspect the data by looking at the bottome 5 rows
maindf.tail()


# In[6]:


# statistical description of numerical columns
maindf.describe()


# In[7]:


# statistical description of non-numerical columns
maindf.describe(include=['object'])


# In[8]:


#review the data element types and empty values
maindf.info()


# In[9]:


#missing data in the following

#Accident DateID	144713	non-null	float64
#Employer Notification DateID	160615	non-null	float64
#Accident State	144713	non-null	object
#Claimant State	160300	non-null	object
#Max Medical Improvement DateID	72580	non-null	float64
#Disability Status	139547	non-null	object
#NCCI BINatureOfLossDescription	146190	non-null	object
#Accident Source Code	71585	non-null	float64
#Accident Type Group	150107	non-null	object

#I would typically clean the missing data (and fill in with median as an example)but have not in this excersise
#due to time constraints


# In[10]:


#lets inspect the floats
flt_df = maindf.select_dtypes(include=['float']).copy()
flt_df.head()


# In[11]:


#lets inspect the objects
obj_df = maindf.select_dtypes(include=['object']).copy()
obj_df.head()



# In[12]:


# in the interest of time, i'm going to drop the floating and object variables from the df as we would
# need to convert the float to numeric and then one hot encode the numeric values 
#to remove any numerical bias in the model 
#if we were to improve the model we would work on filling empty data
#and transforming the floating to numerical catagories then possibly one hot encoding using 


# In[13]:


#while the floating, object and numerical data elements can be cleansed, 
#in the interest of time lets just focus on the boolean catagories for now as they look clean and complete
#lets create a new data frame and name it 'cleandf' with all floating and numeric variables removed

cleandf = maindf.drop(['Accident DateID',
'Claim Setup DateID',
'Report To GB DateID',
'Employer Notification DateID',
'Benefits State',
'Accident State',
'Industry ID',
'Claimant Age',
'Claimant Sex',
'Claimant State',
'Claimant Marital Status',
'Number Dependents',
'Weekly Wage',
'Employment Status Flag',
'RTW Restriction Flag',
'Max Medical Improvement DateID',
'Percent Impairment',
'Post Injury Weekly Wage',
'NCCI Job Code',
'Surgery Flag',
'Disability Status',
'SIC Group',
'NCCI BINatureOfLossDescription',
'Accident Source Code',
'Accident Type Group',
'HCPCS A Codes',
'HCPCS B Codes',
'HCPCS C Codes',
'HCPCS D Codes',
'HCPCS E Codes',
'HCPCS F Codes',
'HCPCS G Codes',
'HCPCS H Codes',
'HCPCS I Codes',
'HCPCS J Codes',
'HCPCS K Codes',
'HCPCS L Codes',
'HCPCS M Codes',
'HCPCS N Codes',
'HCPCS O Codes',
'HCPCS P Codes',
'HCPCS Q Codes',
'HCPCS R Codes',
'HCPCS S Codes',
'HCPCS T Codes',
'HCPCS U Codes',
'HCPCS V Codes',
'HCPCS W Codes',
'HCPCS X Codes',
'HCPCS Y Codes',
'HCPCS Z Codes',
'ICD Group 1',
'ICD Group 2',
'ICD Group 3',
'ICD Group 4',
'ICD Group 5',
'ICD Group 6',
'ICD Group 7',
'ICD Group 8',
'ICD Group 9',
'ICD Group 10',
'ICD Group 11',
'ICD Group 12',
'ICD Group 13',
'ICD Group 14',
'ICD Group 15',
'ICD Group 16',
'ICD Group 17',
'ICD Group 18',
'ICD Group 19',
'ICD Group 20',
'ICD Group 21',
'CPT Category - Anesthesia',
'CPT Category - Eval_Mgmt',
'CPT Category - Medicine',
'CPT Category - Path_Lab',
'CPT Category - Radiology',
'CPT Category - Surgery',
'NDC Class - Benzo',
'NDC Class - Misc (Zolpidem)',
'NDC Class - Muscle Relaxants',
'NDC Class - Stimulants', 'ClaimID'
], axis=1)


# In[14]:


cleandf.head()


# In[15]:


#MACHINE LEARNING ALGORITHM

#We are going to use a classification model, Random Forest

X_train = cleandf.drop("Opiods Used", axis=1)
y_train = cleandf["Opiods Used"]
X_test  = cleandf.copy()


# In[27]:


#Split the the original dataset to train then test using a sample of the original dataset

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# In[28]:


#RandomForest 
Classifier= RandomForestClassifier(n_estimators=200, random_state=10000)
Classifier.fit(X_train, y_train)
y_pred = Classifier.predict(X_test)


# In[31]:


#Validate the accuracy of the model

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[24]:


#Our accuracy is around 90%
#this means out of 100 claimants the model could accurately predict opiod use (or not) 90% of the time


# In[32]:


#Present results merged with input as df
X_test["Opiods Used"] = Classifier.predict(X_test)


# In[22]:


X_test


# In[25]:


#Generate output in .csv format 
X_test.to_csv("opiodusepredictionresults.csv")

