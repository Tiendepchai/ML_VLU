import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("../Data/drug200.csv")
df_dummy = pd.get_dummies(df[['Age','Sex','BP','Cholesterol','Na_to_K']],drop_first=True)
df_dummy.replace({False: 0, True: 1}, inplace=True)

y = df['Drug']
X = df_dummy
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=35)
model = GaussianNB().fit(X_train, y_train)
import streamlit as st
pre_in = st.text_input("YEAR_OLD NATO M/F BP(LOW/NORMAL) Cholesterol(NORMAL/HIGH)", "")
pre = pre_in.split()

Data = {'Age': [int(pre[0])],'Na_to_K': [int(pre[1])] ,'Sex': [pre[2]],'BP' : [pre[3]],'Cholesterol': [pre[4]]}
dff = pd.DataFrame(Data)
df_pre = pd.get_dummies(dff[['Age','Sex','BP','Cholesterol','Na_to_K']])
df_pre.replace({False: 0, True: 1}, inplace=True)
df_pre["BP_NORMAL"] = df_pre["BP_LOW"]
for i in range(len(df_pre["BP_LOW"])):
  df_pre["BP_NORMAL"][i] =  1 if df_pre["BP_LOW"][i] == 0 else 0


st.write(model.predict(df_pre[['Age','Na_to_K', 'Sex_M','BP_LOW', 'BP_NORMAL','Cholesterol_NORMAL']]))
# print()