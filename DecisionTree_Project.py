
# Predict the income of a person 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing required packages for visualization
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('C:\\Users\\Sasi\\Anaconda3\\MyFiles\\DT-forudemy\\adult_dataset.csv')
df.info()

df1 = df[df.workclass == "?"]

#only 5% of the data has missing values, so we will drop the rows with missing values
df = df[df.workclass != "?"]

# look if there are other columns with Null values or ?. 
# Since "?" is a string, we can apply this check only on the categorical columns
#categorical data removes numerical columns from the dataset

df_categorical = df.select_dtypes(include=['object'])
df_categorical.apply(lambda x: x =="?", axis=0).sum()

# as it is less data, we will drop them off
df = df[df.occupation != '?']
df = df[df['native.country'] != '?'] # df.coulmn name did not work here as column name has a dot (.)


# DATA PREPARATION
#For regression, as it processes only numerical data, categorical data was given dummy values of 0, 1 etc
#but thats not needed in DT. However, data encoding (LabelENcoder())is required for categorical data for scikitlearn to process data
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()

# Apply Label Encoder
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

#Drop categorical columns from original df and concatenate with the LabelEncoder values

df = df.drop(df_categorical.columns, axis=1)
df = pd.concat([df, df_categorical], axis=1)

# convert target variable income to categorical
df['income'] = df['income'].astype('category')

# MODEL BUILDING AND EVALUATION

X= df.drop('income', axis=1)
Y = df['income']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state = 99)


#imported Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(x_train, y_train)


# Predict and generate reports
# Making predictions
y_pred = dt_model.predict(x_test)
    
# Printing classification report\n",
print(classification_report(y_test, y_pred))

# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# Putting features
features = list(df.columns[1:])
features


# plotting tree with max_depth=3
dot_data = StringIO()
export_graphviz(dt_model, out_file=dot_data, feature_names=features, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# HYPERPARAMETER TRAINING TO MAKE IT SIMPLER
# WIP
