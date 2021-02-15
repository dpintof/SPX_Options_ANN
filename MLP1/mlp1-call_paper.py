#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras import backend
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[3]:


# Hyperparams
n_units = 400
layers = 4
n_batch = 4096
n_epochs = 10


# In[4]:


df = pd.read_csv('../options-df-sigma.csv')
df = df.dropna(axis=0)
df = df.drop(columns=['date', 'exdate', 'impl_volatility', 'volume', 'open_interest'])
df.strike_price = df.strike_price / 1000
call_df = df[df.cp_flag == 'C'].drop(['cp_flag'], axis=1)
put_df = df[df.cp_flag == 'P'].drop(['cp_flag'], axis=1)


# In[5]:


call_df.head()


# In[6]:


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(['best_bid', 'best_offer'], axis=1),
                                                                        (call_df.best_bid + call_df.best_offer) / 2,
                                                                        test_size=0.01, random_state=42)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(['best_bid', 'best_offer'], axis=1),
                                                                    (put_df.best_bid + put_df.best_offer) / 2,
                                                                    test_size=0.01, random_state=42)


# In[7]:


model = Sequential()
model.add(Dense(n_units, input_dim=call_X_train.shape[1]))
model.add(LeakyReLU())

for _ in range(layers - 1):
    model.add(Dense(n_units))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer=Adam())


# In[8]:


model.summary()


# In[9]:


history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[10]:


model.save('mlp1-call10.h5')


# In[11]:


model.compile(loss='mse', optimizer=Adam(lr=1e-4))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=4096, epochs=n_epochs, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[12]:


model.save('mlp1-call20.h5')


# In[13]:


model.compile(loss='mse', optimizer=Adam(lr=1e-5))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=4096, epochs=10, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[14]:


model.save('mlp1-call30.h5')


# In[15]:


model.compile(loss='mse', optimizer=Adam(lr=1e-6))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=4096, epochs=10, 
                    validation_split = 0.01,
                    callbacks=[TensorBoard()],
                    verbose=1)


# In[25]:


model.save('mlp1-call40.h5')


# In[16]:


call_y_pred = model.predict(call_X_test)


# In[22]:


diff = (call_y_test.values - call_y_pred.reshape(call_y_pred.shape[0]))


# In[24]:


np.mean(np.square(diff))

