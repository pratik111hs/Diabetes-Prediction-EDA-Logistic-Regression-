#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


d = r'C:\Users\Pratik Sonawane\Downloads\diabetes.csv'
df = pd.read_csv(d)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


# Target Variababsl

df['Outcome'].value_counts()


# * 500 patients (about 65%) have Outcome = 0 (no diabetes).
# * 268 patients (about 35%) have Outcome = 1 (diabetes).

# In[9]:


grouped_means = df.groupby('Outcome').mean().T

# Calculate percentage difference
grouped_means['% diff'] = 100 * (grouped_means[1] - grouped_means[0]) / grouped_means[0]

print(grouped_means)


# **High differences (above 25%):**
# 
# * Pregnancies: Diabetic patients have almost 48% more pregnancies on average, suggesting a potential link between pre-pregnancy factors and diabetes risk.
# * Glucose: Blood glucose levels are around 28% higher in the diabetic group, highlighting impaired glucose regulation as a major characteristic of the disease.
# * Insulin: Diabetic patients show a significant 46% increase in average insulin levels, indicating insufficient or ineffective insulin production or utilization.
# * DiabetesPedigreeFunction: A 28% increase in this genetic score suggests a possible hereditary influence on diabetes risk.
# 
# **Moderate differences (around 10-25%):**
# 
# * BMI: Diabetic patients have a 16% higher average BMI, reflecting a potential role of body weight in diabetes development.
# * SkinThickness: Increased skin thickness by 13% could be a marker for insulin resistance or metabolic changes associated with diabetes.
# * Age: There's a 19% difference in average age, with diabetic patients being older on average. This could be due to age-related changes in metabolism or longer exposure to risk factors.
# 
# **Low difference (below 5%):**
# 
# * BloodPressure: While slightly higher in diabetic patients, the difference in average blood pressure is relatively small, indicating it might not be a primary factor for everyone.

# In[10]:


df.hist(bins=10,figsize = (15,10))


# In[11]:


df.corr()['Outcome']


# * A high correlation simply indicates a linear relationship between two variables, but it doesn't necessarily mean one directly causes the other.
# * Features with seemingly weak correlations might still be valuable based on ones understanding of the problem and the underlying biology.

# In[15]:


X = df.drop(columns = ['Outcome'])
Y = df['Outcome']


# In[17]:


scaler = StandardScaler()


# In[19]:


scaler.fit(X)


# In[20]:


standard_data = scaler.transform(X)


# In[22]:


X_scaled = scaler.fit_transform(X)
X_scaled


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,Y,test_size = 0.2,stratify = Y,random_state=42)


# In[27]:


print(X.shape,X_train.shape,X_test.shape)


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[29]:


lr = LogisticRegression()

dtc = DecisionTreeClassifier()

rf = RandomForestClassifier()

svm = SVC()

knn = KNeighborsClassifier()

GBM = GradientBoostingClassifier()


# In[31]:


lr.fit(X_train,y_train)
dtc.fit(X_train,y_train)
rf.fit(X_train,y_train)
svm.fit(X_train,y_train)
knn.fit(X_train,y_train)
GBM.fit(X_train,y_train)


# In[32]:


from sklearn.metrics import classification_report


# In[34]:


def model_evaluation(model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

lr_pred, lr_accuracy = model_evaluation(lr)
dtc_pred, dtc_accuracy = model_evaluation(dtc)
rf_pred, rf_accuracy = model_evaluation(rf)
svm_pred, svm_accuracy = model_evaluation(svm)
knn_pred, knn_accuracy = model_evaluation(knn)
GBM_pred, GBM_accuracy = model_evaluation(GBM)

print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
print(f"Decision Tree accuracy: {dtc_accuracy:.4f}")
print(f"Random Forest accuracy: {rf_accuracy:.4f}")
print(f"Support Vector Machine accuracy: {svm_accuracy:.4f}")
print(f"K-Nearest Neighbors accuracy: {knn_accuracy:.4f}")
print(f"Gradient Boosting Model accuracy: {GBM_accuracy:.4f}")



# rf and GBM giving same score , preparing classification report and model with best recall score will be best for our project

# In[39]:


from sklearn.metrics import classification_report

rf_report = classification_report(y_test, rf.predict(X_test))
print('rf :',rf_report)

GBM_report = classification_report(y_test, GBM.predict(X_test))
print('GBM :',GBM_report)


# In[40]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[45]:


lrcv = cross_val_score(lr,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('lr :', lrcv)
dtccv = cross_val_score(dtc,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('dtccv :',dtccv)
rfcv = cross_val_score(rf,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('rfcv :',rfcv)
svmcv = cross_val_score(svm,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('svmcv :', svmcv)
knncv = cross_val_score(knn,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('knncv :', knncv)
GBMcv = cross_val_score(GBM,X_scaled,Y,cv = StratifiedKFold(n_splits=5)).mean()
print('GBMcv :', GBMcv)


# ## Predictive System using SVM Model

# In[46]:


X.columns


# In[48]:


Pregnancies = int(input('Enter the Pregnancies =',))
Glucose = float(input('Enter the Glucose =',))
BloodPressure = float(input('Enter the BloodPressure =',))
SkinThickness = float(input('Enter the SkinThickness =',))
Insulin = float(input('Enter the Insulin =',))
BMI = float(input('Enter the BMI =',))
DiabetesPedigreeFunction = float(input('Enter the DiabetesPedigreeFunction =',))
Age = float(input('Enter the Age =',))

patient_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                BMI,DiabetesPedigreeFunction,Age]
patient_data


# In[49]:


input_array = np.asarray(patient_data)
input_array


# In[50]:


# Reshape in 2D
input_data = input_array.reshape(1,-1)
input_data


# In[51]:


std_data = scaler.transform(input_data)
std_data


# In[52]:


prediction = svm.predict(std_data)
prediction


# In[53]:


prediction[0]


# **``Model has accurately predicted above pateint data with no diabetes(0)``**

# ## Prediction System

# In[57]:


import warnings
warnings.filterwarnings('ignore')

Pregnancies = int(input('Enter the Pregnancies =',))
Glucose = float(input('Enter the Glucose =',))
BloodPressure = float(input('Enter the BloodPressure =',))
SkinThickness = float(input('Enter the SkinThickness =',))
Insulin = float(input('Enter the Insulin =',))
BMI = float(input('Enter the BMI =',))
DiabetesPedigreeFunction = float(input('Enter the DiabetesPedigreeFunction =',))
Age = float(input('Enter the Age =',))

patient_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                BMI,DiabetesPedigreeFunction,Age]

input_array = np.asarray(patient_data)
input_data = input_array.reshape(1,-1)
std_data = scaler.transform(input_data)
prediction = svm.predict(std_data)

print('***************Model Predictions*******************')
if (prediction[0] ==0):
    print('Result - Negative')
    print('The person is Not Diabetic')
else:
    print('Result - Positive')
    print('The Patient is Diabetic')
    

