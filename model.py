#Average test accuracy: approx 99%

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout

path = 'C:/Users/psnit/Desktop/dataset'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
            
accuracy=[]

for f in files: #to access all files in dataset folder
    
    df = pd.read_csv(f, na_filter= False)#nafilter is used to remove nan values
    df = df.sample(frac=1).reset_index(drop=True)#shuffling dataset everytime
    total_rows=len(df.index)
    
    '''
    for i in range(0,total_rows):
        s=df["WindGustDir"][i]
        s=s.replace('N','1')
        s=s.replace('S','2')
        s=s.replace('E','3')
        s=s.replace('W','4')
        df["WindGustDir"][i]=s
        s4=df["WindDir9am"][i]
        s4=s4.replace('N','1')
        s4=s4.replace('S','2')
        s4=s4.replace('E','3')
        s4=s4.replace('W','4')
        df["WindDir9am"][i]=s4
        s1=df["WindDir3pm"][i]
        s1=s1.replace('N','1')
        s1=s1.replace('S','2')
        s1=s1.replace('E','3')
        s1=s1.replace('W','4')
        df["WindDir3pm"][i]=s1
        s2=df["RainToday"][i]
        s2=s2.replace("Yes",'1')
        s2=s2.replace("No",'0')
        df["RainToday"][i]=s2
        s3=df["RainTomorrow"][i]
        s3=s3.replace("Yes",'1')
        s3=s3.replace("No",'0')
        df["RainTomorrow"][i]=s3
    '''
    
    df['WindGustDir'] = pd.Categorical(df['WindGustDir'])
    df['WindGustDir'] = df.WindGustDir.cat.codes
    df['WindDir9am'] = pd.Categorical(df['WindDir9am'])
    df['WindDir9am'] = df.WindDir9am.cat.codes
    df['WindDir3pm'] = pd.Categorical(df['WindDir3pm'])                
    df['WindDir3pm'] = df.WindDir3pm.cat.codes
    df['RainToday'] = pd.Categorical(df['RainToday'])
    df['RainToday'] = df.RainToday.cat.codes
    df['RainTomorrow'] = pd.Categorical(df['RainTomorrow'])
    df['RainTomorrow'] = df.RainTomorrow.cat.codes
    df[['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']] = df[['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']].astype(float) #converting all columns to float
    
    X = df.iloc[:, 2:23].values
    Y = df.iloc[:, 23].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 2) #to split into 80% training and 20% testing
    
    model=Sequential()
    
    model.add(Dense(100,activation='sigmoid',input_dim=21,kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.30,noise_shape=None,seed=None))
    
    model.add(Dense(100,activation='sigmoid',kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.30,noise_shape=None,seed=None))
    
    model.add(Dense(100,activation='sigmoid',kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.30,noise_shape=None,seed=None))
   
    model.add(Dense(1,activation='sigmoid',kernel_initializer='normal'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.summary()
    model_output=model.fit(x_train,y_train,epochs=500,batch_size=16,verbose=1,validation_data=(x_test,y_test),)
    #print('Training Accuracy : ', np.mean(model_output.history["acc"]))
    #print('Validation Accuracy : ', np.mean(model_output.history["val_acc"]))
    
    accuracy.append(np.mean(model_output.history["acc"]))

test_acc=0

for i in range(0,49):
    test_acc=test_acc+accuracy[i]

test_acc=test_acc/49
print("Average test accuracy of all cities is : ",test_acc)
#Average test accuracy: approx 99%
    