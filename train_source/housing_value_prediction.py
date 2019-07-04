#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()


# In[3]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[4]:


housing = load_housing_data()
housing.head()


# In[5]:


housing.info()


# In[6]:


# total bedrooms만 20433개라서 전처리가 필요함
# ocean_proximity 빼고 다 숫자형임


# In[7]:


housing.describe() # 숫자형 특성의 요약정보


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


# median income이 달러단위가 아님 --> 상한, 하한이 설정되어있음 데이터 처리한것임
# 중간 주택연도와 중간 주택가격(레이블이 될 값이라서 클라이언트와 검토필요) 도 최댓값 최솟값 한정시킴
# 각 특성들의 분포가 너무 다름



# In[ ]:




