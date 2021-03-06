# -*- coding: utf-8 -*-
"""911-visualization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fK5U3rdjK12wlxZ2glJ-xijtvBgwIrC2
"""
# 머신러닝은 아니고.. 그냥 data visualization
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

df = pd.read_csv('911.csv')

df.info()
df.head()

#Lets see top 5 zip and townships that have called the 911.
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)

# Lets see how many unique titles are there in the datasets
df['title'].nunique() # 유니크한 reason 갯수 100개

# 신고유형으로 'reason'  새로 생성
df['reason'] = df['title'].astype(str).apply(lambda title: title.split(':')[0])
df['reason'].head(4)
df['reason'].value_counts()

sns.countplot(x = 'reason', data = df) # EMS가 가장 많음

# 시간 값 간단하게 바꾸기
# str타입이므로 datetime 형식으로 바꾸기
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])

# 시간 정보 컬럼 생성
df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)
df['month'] = df['timeStamp'].apply(lambda time:time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)
df.head(1)

# 요일로 바꾸기
dmap = {0:'mon', 1:'tue',2:'wed', 3:'thu', 4:'fri', 5:'sat', 6:'sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head(1)

# 무슨요일에 무슨 사건이 발생하는지
sns.countplot(data = df, x='Day of Week', hue = 'reason', palette = 'viridis')
plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0)

# 무슨달에 무슨 사건이 발생하는지
sns.countplot(data = df, x = 'month', hue = 'reason', palette = 'viridis')
plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0)

df['date'] = df['timeStamp'].apply(lambda t:t.date())
#Group by the data as per date and plot it.
fig = plt.figure(figsize=(10,5))
df.groupby('date').count()['lat'].plot()
plt.tight_layout()

#For Traffic
fig = plt.figure(figsize=(10,5))
df[df['reason']=='Traffic'].groupby('date').count()['lat'].plot()
plt.tight_layout()
plt.title('Traffic')

#For EMS
fig = plt.figure(figsize=(10,5))
df[df['reason']=='EMS'].groupby('date').count()['lat'].plot()
plt.tight_layout()
plt.title('EMS')

#For EMS
fig = plt.figure(figsize=(10,5))
df[df['reason']=='Fire'].groupby('date').count()['lat'].plot()
plt.tight_layout()
plt.title('Fire')

dayHour = df.groupby(by = ['Day of Week', 'Hour']).count()['reason'].unstack()
dayHour # 각 요일 시간 별 발생 건수

#Clustermap for the same
fig = plt.figure(figsize = (10,7))
sns.clustermap(dayHour, cmap = 'coolwarm')

