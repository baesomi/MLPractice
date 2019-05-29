# -*- coding: utf-8 -*-
"""black friday.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VZrh_MzSgSBlp-YzT3c8IBxFRY5TfTeH
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline


sns.set(style='darkgrid')
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv('BlackFriday.csv')
df.head(5)

print(df.info())
print('shape: ', df.shape)

# Missing values : product_category_2,3 이 null 많음
total_miss = df.isnull().sum()
perc_miss = total_miss / df.isnull().count() * 100

missing_data = pd.DataFrame({'Total missing': total_miss, '% missing': perc_miss})

missing_data.sort_values(by='Total missing', ascending=False).head(3)

# Unique values

print('각 feature의 unique한 값 : \n')

for i in df.columns:
    print(i, ':', df[i].nunique())

# 제품 정보
print('제품 수:', df['Product_ID'].nunique())
print('제품군 수:', df['Product_Category_1'].unique().max())
print('최고가, 최저가:', df['Purchase'].max(), ',', df['Purchase'].min())

# 구매자 정보
print('구매자 수:', df['User_ID'].nunique())
print('거주기간:', df['Stay_In_Current_City_Years'].unique())
print('구매자 연령대:', df['Age'].unique())

# 성별
count_f = df[df['Gender'] == 'F'].count()[0]
count_m = df[df['Gender'] == 'M'].count()[0]

# 남 녀 4배 차이
print('여성 : ', count_f)  # 132197
print('남성 : ', count_m)  # 405380

print('여성 1인당 평균 구매액 : ', round(df[df['Gender'] == 'F']['Purchase'].sum() / count_f, 3))
print('남성 1인당 평균 구매액 : ', round(df[df['Gender'] == 'M']['Purchase'].sum() / count_m, 3))

plt.pie(df.groupby('Gender')['Product_ID'].nunique(), labels=['Male', 'Female'], shadow=True, autopct='%1.1f%%',
        colors=['steelblue', 'red'])
plt.title('Unique Item Purchases by Gender')
plt.show()  # 둘이 비슷하긴한데 여자가 약간 더 다양한 구매를 함

# 성별당 각 제품군 구매율
gb_gender_m = df[df['Gender'] == 'M'][['Product_Category_1', 'Gender']].groupby(by='Product_Category_1').count()
gb_gender_f = df[df['Gender'] == 'F'][['Product_Category_1', 'Gender']].groupby(by='Product_Category_1').count()

# 남녀 비율
cat_bygender = pd.concat([gb_gender_m, gb_gender_f], axis=1)
cat_bygender.columns = ['M ratio', 'F ratio']

# 비율 반영하도록 조정
cat_bygender['M ratio'] = cat_bygender['M ratio'] / df[df['Gender'] == 'M'].count()[0]
cat_bygender['F ratio'] = cat_bygender['F ratio'] / df[df['Gender'] == 'F'].count()[0]

# 성별에 따라 살 가능성
cat_bygender['Likelihood (M/F)'] = cat_bygender['M ratio'] / cat_bygender['F ratio']

cat_bygender['Total Ratio'] = cat_bygender['M ratio'] + cat_bygender['F ratio']

cat_bygender.sort_values(by='Likelihood (M/F)', ascending=False)

# 연령대

# 각 값에 대해 인코딩
df['Age_Encoded'] = df['Age'].map({'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6})

# 연령대 별로 구입하는 물품 ID
prod_byage = df.groupby('Age').nunique()['Product_ID']

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax = ax.ravel()

sns.countplot(df['Age'].sort_values(), ax=ax[0], palette="Blues_d")
ax[0].set_xlabel('Age Group')
ax[0].set_title('Age Group Distribution')  # 구매자 연령대별 분포도 max:26-35

sns.barplot(x=prod_byage.index, y=prod_byage.values, ax=ax[1], palette="Blues_d")
ax[1].set_xlabel('Age Group')
ax[1].set_title('Unique Products by Age')  # 연령별 얼마나 다양한 제품을 사는지 -> 제품 구매의 분포는 수량면에서 연령대별로 크게 다르지 않음

plt.show()

# 연령별 구매금액은?
spent_byage = df.groupby(by='Age').sum()['Purchase']  # 연령별 구매금액 합
plt.figure(figsize=(12, 6))

sns.barplot(x=spent_byage.index, y=spent_byage.values, palette="Blues_d")
plt.title('Mean Purchases per Age Group')
plt.show()  # 구매 분포는 비슷하지만 구매금액은 압도적으로 26-35가 높음

# 직업? 같은데 너무 많아서 쓸게 음슴
plt.figure(figsize=(12, 6))
sns.countplot(df['Occupation'])
plt.title('Occupation Distribution')
plt.show()

# Products
plt.figure(figsize=(12, 6))
prod_by_cat = df.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index, y=prod_by_cat.values, palette="Blues_d")
plt.title('Number of Unique Items per Category')  # 카테고리별 unique한 item 갯수
plt.show()

# 그럼 제품군별 구매량은?
category = []
mean_purchase = []

for i in df['Product_Category_1'].unique():
    category.append(i)

category.sort()

for e in category:
    mean_purchase.append(df[df['Product_Category_1'] == e]['Purchase'].mean())

plt.figure(figsize=(12, 6))

sns.barplot(x=category, y=mean_purchase)
plt.title('Mean of the Purchases per Category')
plt.xlabel('Product Category')
plt.ylabel('Mean Purchase')
plt.show()  # 1,5,8이 제일 인기가 많지만 제일 구매액이 높은 제품은 10,7 로 다른양상을 보임
