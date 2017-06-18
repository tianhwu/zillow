
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import os
print(os.getcwd())


# In[2]:

train = pd.read_csv('../input/train_2016_v2.csv')


# In[3]:

prop = pd.read_csv('../input/properties_2016.csv')


# In[4]:

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)


# In[5]:

df_train = train.merge(prop, how='left', on='parcelid')


# In[6]:

df_train.info()


# In[7]:

x_train = df_train.drop(['parcelid', 
                         'logerror', 
                         'transactiondate', 
                         'propertyzoningdesc', 
                         'propertycountylandusecode'
                         ], axis=1)

train_columns = x_train.columns

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


# In[8]:

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


# In[9]:

#del df_train; gc.collect()
split = int(round(.84*df_train.shape[0],0))

split


# In[10]:


x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)


# In[11]:

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)


# In[12]:

#stolen tuning from someone elses notebok
params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['num_leaves'] = 31
params['min_data'] = 400
params['min_data_in_leaf'] = 450
params['min_hessian'] = 1
params['feature_fraction'] = .5


# In[14]:

watchlist = [d_valid]


# In[15]:

clf = lgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=200)


# In[16]:


print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()


# In[17]:

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)


# In[18]:


print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})


# In[ ]:

p_test = clf.predict(x_test)


# In[ ]:


p_test = 0.97*p_test + 0.03*0.011

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgbm_starter.csv', index=False, float_format='%.4f')

