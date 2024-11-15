#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys

required_packages = [
    'tensorflow', 'pandas', 'matplotlib', 'azureml-core', 
    'scikit-learn', 'azure-ai-ml', 'azure-identity', 'azureml-sdk'
]

def install_packages(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

# In[1]:

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from azureml.core import Run
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Model
from azureml.core import Dataset

# In[2]:
config_path = 'config.json'
ws = Workspace.from_config(path=config_path)
run = Run.get_context()

class AzureMLCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        run.log('train_accuracy', logs['accuracy'])
        run.log('validation_accuracy', logs['val_accuracy'])
        run.log('train_loss', logs['loss'])
        run.log('validation_loss', logs['val_loss'])

dataset = Dataset.get_by_name(ws, name="healthcare-heart-disease", version="1")

local_path = dataset.download(target_path='./data', overwrite=True)

dataset_file_path = local_path[0]  

dataSet = pd.read_csv(dataset_file_path)


# In[3]:


gender_target_counts = dataSet.groupby(['sex', 'target']).size().unstack(fill_value=0)
gender_target_counts.plot(kind='bar', stacked=True, color=['gold', 'darkblue'])

plt.title('Heart Attack Risk by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], rotation=0)
plt.legend(title='Heart Attack Risk', labels=['Low Risk', 'High Risk'])

plt.tight_layout()
plt.show()


# In[4]:


high_risk_df = dataSet[dataSet['target'] == 1]
plt.hist(high_risk_df['age'], bins=10, color='gold', edgecolor='darkblue')

plt.title('Age Distribution of High Heart Attack Risk Individuals')
plt.xlabel('Age')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[5]:


X=dataSet.drop(columns=['target'])  
Y=dataSet['target']

scaler=StandardScaler()
XScaled=scaler.fit_transform(X)

XScaled=pd.DataFrame(XScaled,columns=X.columns)
scaledDataSet=pd.concat([XScaled,Y], axis=1)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# In[7]:


dataX=scaledDataSet.drop(columns=['target'])  
dataY=scaledDataSet['target'] 

trainX,testX,trainY,testY=train_test_split(dataX, dataY, test_size=0.2, random_state=42)


# In[8]:


model=Sequential([
    Input(shape=(trainX.shape[1],)),
    Dense(128,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1,activation='sigmoid'),
])


# In[9]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(trainX,trainY, epochs=100, batch_size=32, validation_data=(testX,testY),callbacks=[AzureMLCallback()])

loss, accuracy = model.evaluate(testX,testY)
print(f'Accuracy: {accuracy*100:.2f}%')

# In[10]:

model.save("cardiopredict_model.h5")
model=Model.register(workspace=ws, model_path="cardiopredict_model.h5", model_name="cardiopredict_model_model")
scaler_model = Model.register(workspace=ws, model_path="scaler.pkl", model_name="scaler_model")





