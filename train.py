import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
df=pd.read_csv('DATA.csv')

train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

df_labels=train_set['prob'].copy()

df1=train_set.drop('prob',axis=1)

clf=LogisticRegression()
clf.fit(df1,df_labels)

input=[100,1,22,1,1]
prob=clf.predict_proba([input])[0][1]
file=open('model.pkl','wb')
pickle.dump(clf,file)
file.close()