#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from keras import backend
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


# Hyperparams
n_units = 400
layers = 4
n_batch = 1024
n_epochs = 200


# In[3]:


df = pd.read_csv('options-df-sigma.csv')
df = df.dropna(axis=0)
df = df.drop(columns=['date', 'exdate', 'impl_volatility'])
df.strike_price = df.strike_price / 1000
call_df = df[df.cp_flag == 'C'].drop(['cp_flag'], axis=1)
put_df = df[df.cp_flag == 'P'].drop(['cp_flag'], axis=1)


# In[4]:


call_df.head()


# In[5]:


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(['best_bid', 'best_offer'], axis=1),
                                                                        (call_df.best_bid + call_df.best_offer) / 2,
                                                                        test_size=0.01, random_state=42)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(['best_bid', 'best_offer'], axis=1),
                                                                    (put_df.best_bid + put_df.best_offer) / 2,
                                                                    test_size=0.01, random_state=42)


# In[8]:


model = Sequential()
model.add(Dense(n_units, input_dim=call_X_train.shape[1]))
model.add(LeakyReLU())

for _ in range(layers - 1):
    model.add(Dense(n_units))
    model.add(LeakyReLU())

model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer=Adam(lr=1e-5))


# In[9]:


model.summary()


# In[11]:


history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[12]:


model.save('mlp1-100.h5')


# In[13]:


call_y_pred = model.predict(call_X_test)


# In[22]:


'test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred, call_y_pred.shape[0])))


# # Now beyond 100 epochs using 1e-6 lr

# In[23]:


model.compile(loss='mse', optimizer=Adam(lr=1e-6))


# In[24]:


history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=20, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[25]:


model.save('mlp1-110.h5')


# In[26]:


call_y_pred2 = model.predict(call_X_test)
'test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred2, call_y_pred2.shape[0])))


# # Now 1e-7 lr

# In[27]:


model.compile(loss='mse', optimizer=Adam(lr=1e-7))


# In[28]:


history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[29]:


model.save('mlp1-115.h5')
call_y_pred3 = model.predict(call_X_test)
'test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred3, call_y_pred3.shape[0])))


# # Now 1e-8 lr

# In[30]:


model.compile(loss='mse', optimizer=Adam(lr=1e-8))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[31]:


model.save('mlp1-120.h5')
call_y_pred4 = model.predict(call_X_test)
'test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred4, call_y_pred4.shape[0])))


# In[ ]:




