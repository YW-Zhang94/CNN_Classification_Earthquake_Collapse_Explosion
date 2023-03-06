import numpy as np
import matplotlib.pyplot as plt
import os
import obspy
import csv
from obspy import read
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime
from pathlib import Path
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, ZeroPadding1D
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import random

models_rout='../Models/refined/'  

#read data
X_eq, Y_eq, nev_eq = [], [], []
X_cl, Y_cl, nev_cl = [], [], []
X_ep, Y_ep, nev_ep = [], [], []

x_eq, y_eq, nneq = [], [], []
x_cp, y_cp, nncp = [], [], []
x_ep, y_ep, nnep = [], [], []


with open('../Data/info.list') as l:
    for line in l:
        rt_list=str(line.split()[0])
        print(rt_list)
        remain_01=0
        with open('../Data/'+rt_list) as e:
            for neq in e:
                if neq.split()[1]=='Sg' or neq.split()[1]=='SMZ': continue
                if float(neq.split()[4])<1.5: continue
                #print(neq.split()[0])
                s=str(neq.split()[0])
                nst=s.split('.')[1]    #station name
                nnet=s.split('.')[0]   #network name
                #P arrival time
                time=UTCDateTime(str(neq.split()[2])+' '+str(neq.split()[3]))-8*60*60
                if nst=='PX': continue
                
                sz=read('../Data/SAC/'+rt_list[5:25]+'_'+nst+'_'+nnet+'_z.sac')[0].data
                se=read('../Data/SAC/'+rt_list[5:25]+'_'+nst+'_'+nnet+'_e.sac')[0].data
                sn=read('../Data/SAC/'+rt_list[5:25]+'_'+nst+'_'+nnet+'_n.sac')[0].data
                
                if  max(max(sz),max(se),max(sn))==0: continue
                trs=[]
                for i in range(5000):
                    trs.append(np.array([sz[i], se[i], sn[i]]))
                    
                ncheck=0    #check if event has different label with original label
                remain_01=1 #check if event is remained after manually checking
                with open('../Data/list.error') as le:
                    for line_le in le:
                        le_neq=str(line_le.split()[1])
                        le_type=str(line_le.split()[5])
                        le_01=int(line_le.split()[6])
                        if le_neq == rt_list[5:20]:
                            #print(le_neq)
                            if le_01 == 0:
                                remain_01=0
                                ncheck=1
                                break
                            if le_type == 'EQ':
                                c=[1,0,0]
                                X_eq.append(np.array(trs))
                                Y_eq.append(c)
                                nev_eq.append(rt_list[5:25]+'_'+s)
                                
                            if le_type == 'CP':
                                c=[0,1,0]
                                X_cl.append(np.array(trs))
                                Y_cl.append(c)
                                nev_cl.append(rt_list[5:25]+'_'+s)
                                
                            if le_type == 'EP':
                                c=[0,0,1]
                                X_ep.append(np.array(trs))
                                Y_ep.append(c)
                                nev_ep.append(rt_list[5:25]+'_'+s)
                            
                            ncheck=1    
                            break
                if ncheck==1: continue
                
                if str(neq.split()[5]) == 'earthquake': 
                    c=[1,0,0]
                    X_eq.append(np.array(trs))
                    Y_eq.append(c)
                    nev_eq.append(rt_list[5:25]+'_'+s)
                if str(neq.split()[5]) == 'collapse': 
                    c=[0,1,0]
                    X_cl.append(np.array(trs))
                    Y_cl.append(c)
                    nev_cl.append(rt_list[5:25]+'_'+s)
                if str(neq.split()[5]) == 'explode': 
                    c=[0,0,1]
                    X_ep.append(np.array(trs))
                    Y_ep.append(c)
                    nev_ep.append(rt_list[5:25]+'_'+s)
                
            if remain_01 == 1:
                if c[0]==1 and len(nev_eq)!=0: 
                    x_eq.append(X_eq)
                    y_eq.append(Y_eq)
                    nneq.append(nev_eq)
                    X_eq, Y_eq, nev_eq = [], [], []
                if c[1]==1 and len(nev_cl)!=0: 
                    x_cp.append(X_cl)
                    y_cp.append(Y_cl)
                    nncp.append(nev_cl)
                    X_cl, Y_cl, nev_cl = [], [], []
                if c[2]==1 and len(nev_ep)!=0: 
                    x_ep.append(X_ep)
                    y_ep.append(Y_ep)
                    nnep.append(nev_ep)
                    X_ep, Y_ep, nev_ep = [], [], []
            
            
            print(np.array(x_eq).shape, np.array(y_eq).shape)
            print(np.array(x_cp).shape, np.array(y_cp).shape)
            print(np.array(x_ep).shape, np.array(y_ep).shape)
            
#seperate data into 10 groups
X_eq, Y_eq, neq = [], [], []
X_cp, Y_cp, ncp = [], [], []
X_ep, Y_ep, nep = [], [], []

for i in range(10):
    tmp_X, tmp_Y, tmp_nev = [], [], []
    for ii in range(int(len(x_eq)*i/10),int(len(x_eq)*(i+1)/10)):
        for iii in range(len(x_eq[ii])):
            tmp_X.append(np.array(x_eq[ii][iii]))
            tmp_Y.append(np.array(y_eq[ii][iii]))
            tmp_nev.append(np.array(nneq[ii][iii]))
    X_eq.append(np.array(tmp_X))
    Y_eq.append(np.array(tmp_Y))
    neq.append(np.array(tmp_nev))
    
    tmp_X, tmp_Y, tmp_nev = [], [], []
    for ii in range(int(len(x_cp)*i/10),int(len(x_cp)*(i+1)/10)):
        for iii in range(len(x_cp[ii])):
            tmp_X.append(np.array(x_cp[ii][iii]))
            tmp_Y.append(np.array(y_cp[ii][iii]))
            tmp_nev.append(np.array(nncp[ii][iii]))
    X_cp.append(np.array(tmp_X))
    Y_cp.append(np.array(tmp_Y))
    ncp.append(np.array(tmp_nev))
    
    tmp_X, tmp_Y, tmp_nev = [], [], []
    for ii in range(int(len(x_ep)*i/10),int(len(x_ep)*(i+1)/10)):
        for iii in range(len(x_ep[ii])):
            tmp_X.append(np.array(x_ep[ii][iii]))
            tmp_Y.append(np.array(y_ep[ii][iii]))
            tmp_nev.append(np.array(nnep[ii][iii]))
    X_ep.append(np.array(tmp_X))
    Y_ep.append(np.array(tmp_Y))
    nep.append(np.array(tmp_nev))

x_eq=X_eq
y_eq=Y_eq

x_cp=X_cp
y_cp=Y_cp

x_ep=X_ep
y_ep=Y_ep

for i in range(10):
    print(str(i+1).zfill(3))
    
    #seperate training and testing datasets
    X_train, Y_train, nev_train = [], [], []
    for ii in range(10):
        if ii == i: 
            X_test=list(np.array(x_eq[ii]))
            Y_test=list(np.array(y_eq[ii]))
            nev_test=list(np.array(neq[ii]))

            X_test.extend(np.array(x_cp[ii]))
            Y_test.extend(np.array(y_cp[ii]))
            nev_test.extend(np.array(ncp[ii]))

            X_test.extend(np.array(x_ep[ii]))
            Y_test.extend(np.array(y_ep[ii]))
            nev_test.extend(np.array(nep[ii]))
    
            X_test=np.array(X_test)
            Y_test=np.array(Y_test)
            nev_test=np.array(nev_test)
            
            continue
            
        for iii in range(len(x_eq[ii])):
            X_train.append(x_eq[ii][iii])
            Y_train.append(y_eq[ii][iii])
            nev_train.append(neq[ii][iii])
            
        for iii in range(len(x_cp[ii])):
            X_train.append(x_cp[ii][iii])
            Y_train.append(y_cp[ii][iii])
            nev_train.append(ncp[ii][iii])
            
        for iii in range(len(x_ep[ii])):
            X_train.append(x_ep[ii][iii])
            Y_train.append(y_ep[ii][iii])
            nev_train.append(nep[ii][iii])
            
    print(np.array(X_train).shape, np.array(Y_train).shape, np.array(nev_train).shape)
    print(np.array(X_test).shape, np.array(Y_test).shape, np.array(nev_test).shape)
    
    #random shuffle training dataset
    X_train_r, Y_train_r, nev_train_r=[], [], []
    index = [ii for ii in range(len(X_train))]
    random.shuffle(index)
    for ii in index:
        X_train_r.append(np.array(X_train[ii]))
        Y_train_r.append(np.array(Y_train[ii]))
        nev_train_r.append(np.array(nev_train[ii]))
        
    X_train_r=np.array(X_train_r)
    Y_train_r=np.array(Y_train_r)
    
    class_weight={0:1., 1:8., 2:2.5}
    
    #model name for saving
    nmodel='CNN_'+str(i).zfill(2)+'.h5'
    
    
    input_shape = (5000,3)
    batch_size=64
    epochs=32

    model = Sequential()
    
    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    
    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Conv1D(kernel_size=(3),filters=64,
              input_shape = input_shape,
              strides = (2),
              padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Flatten())

    model.add(Dense(units = (128)))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(units = (3),
              activation='softmax'))
    print(model.summary())

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    print('*-----------------')
    
    models_rout=models_rout+nmodel
    
    model.load_weights(models_rout)
    
    #show results
    n_eq=0
    result = model.predict(np.array(x_eq[i]))
    for ii in range(len(x_eq[i])):
        if max(result[ii]) == result[ii][0]: n_eq+=1
    print('earthquake accuracy:', n_eq, len(x_eq[i]), n_eq/len(x_eq[i]))

    n_cl=0
    result = model.predict(np.array(x_cp[i]))
    for ii in range(len(x_cp[i])):
        if max(result[ii]) == result[ii][1]: n_cl+=1
    print('collapse accuracy:', n_cl, len(x_cp[i]), n_cl/len(x_cp[i]))

    n_ep=0
    result = model.predict(np.array(x_ep[i]))
    for ii in range(len(x_ep[i])):
        if max(result[ii]) == result[ii][2]: n_ep+=1
    print('explode accuracy:', n_ep, len(x_ep[i]), n_ep/len(x_ep[i]))
    
    
    #save results
    nlist=str(i).zfill(2)+'.list'
    
    with open(nlist,'wt') as el:    
        result = model.predict(np.array(x_eq[i]))
        for ii in range(len(x_eq[i])):
            #print('earthquake '+nev_eq_test[i],result[i])
            el.write('EQ %-40s %.4f %.4f %.4f \n' % (neq[i][ii], result[ii][0], result[ii][1], result[ii][2]))
            #print(result[i])

        result = model.predict(np.array(x_cp[i]))
        for ii in range(len(x_cp[i])):
            #print('collapse '+nev_cl_test[i],result[i])
            el.write('CP %-40s %.4f %.4f %.4f \n' % (ncp[i][ii], result[ii][0], result[ii][1], result[ii][2]))
            #print(result[i])

        result = model.predict(np.array(x_ep[i]))
        
        
        
        for ii in range(len(x_ep[i])):
            #print('explode '+nev_ep_test[i],result[i])
            el.write('EP %-40s %.4f %.4f %.4f \n' % (nep[i][ii], result[ii][0], result[ii][1], result[ii][2]))
                
