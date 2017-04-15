# -*- coding: utf-8 -*-
"""
Created on Thu April 13 16:25:19 2017

@author: lingyu
"""

import numpy as np
import csv
import sys
import os
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


seed =7
np.random.seed(seed)
NB_EPOCHS=500
BATCH_SIZE=200
VERBOSE=0
Nb_fm=32# number of feature maps
k=3 # size of kernel
m=3 # stride
DROP_OUT=0.5
num_classes=1


def main():

    def CRNN_model(Nb_fm,k,m,DROP_OUT,num_classes):
        # create model
        model=Sequential()
        model.add(Convolution2D(Nb_fm,k,m,input_shape=(1,100,5),border_mode='same', activation='relu'))
	model.add(Dropout(DROP_OUT))      
        model.add(Convolution2D(Nb_fm,k,m,border_mode='same', activation='relu'))
	model.add(Dropout(DROP_OUT))
	model.add(Reshape((100,5*Nb_fm)))
	model.add(LSTM(32,input_shape=(100,5*Nb_fm),dropout_W=DROP_OUT,dropout_U=DROP_OUT))
        model.add(Dropout(DROP_OUT))
        model.add(Dense(100,init='uniform',activation='relu'))
        model.add(Dense(20,init='uniform',activation='relu'))
        model.add(Dense(num_classes,init='uniform',activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    # setting path
    first_path='.../Data'
    base_filename_3_1='_rc_x'
    base_filename_3_2='_rc_y'
    base_filename_4_1='_train'
    base_filename_4_2='_test'
    filename_suffix_out='csv'
    
    # load cell types
    print('load cell types...')
    cellType=np.loadtxt('.../cellType.txt',delimiter='\t',dtype=str)
    nb_split=len(cellType) # the number of classes
    auc=[]
    for i in range(0,nb_split):
        dir_name=cellType[i]
        
	# get the path of data file
        save_path_x_train=os.path.join(first_path, dir_name + base_filename_3_1 + base_filename_4_1 + '.' + filename_suffix_out)
        save_path_x_test=os.path.join(first_path, dir_name + base_filename_3_1 + base_filename_4_2 + '.' + filename_suffix_out)
        save_path_y_train=os.path.join(first_path, dir_name + base_filename_3_2 + base_filename_4_1 + '.' + filename_suffix_out)
        save_path_y_test=os.path.join(first_path, dir_name + base_filename_3_2 + base_filename_4_2 + '.' + filename_suffix_out)
        
	# load data
        print("Loading data...",dir_name)
        X_train = np.loadtxt(save_path_x_train, 
                             delimiter = ",", skiprows = 1)
        X_test  = np.loadtxt(save_path_x_test, 
                             delimiter = ",", skiprows = 1)    
        y_train = np.loadtxt(save_path_y_train, 
                             delimiter = ",", skiprows = 1)
        y_test = np.loadtxt(save_path_y_test, 
                             delimiter = ",", skiprows = 1)
        print "All files loaded. Preprocessing..."

        # remove the first column(Gene Id)
        X_train = X_train[:,1:] 
        X_test  = X_test[:,1:]   
        y_train = y_train[:,1:]
        y_test = y_test[:,1:]
    
        # Every 100 rows correspond to one gene.
        # Extract all 100-row-blocks into a list using np.split.
        num_genes_train = X_train.shape[0] / 100
        num_genes_test  = X_test.shape[0] / 100
    
        print("Train / test data has %d / %d genes." % \
              (num_genes_train, num_genes_test))
        X_train = np.split(X_train, num_genes_train)
        X_test  = np.split(X_test, num_genes_test)
        
        # convert data from list to array
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test  = np.array(X_test)
        y_train = np.ravel(y_train)
        y_test = np.array(y_test)
        
        # reshape--(samples,channels,rows,cols)
        X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
        X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))

        # convert data from list to array
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test  = np.array(X_test)
        y_train = np.ravel(y_train)
        y_test  = np.array(y_test)

        y_train = y_train[:].astype('int')
        y_test = y_test[:].astype('int')
        
        print("X_train shape is %s" % str(X_train.shape))    
        print("y_train shape is %s" % str(y_train.shape))
        print("X_test shape is %s" % str(X_test.shape))    
        print('Data preprocessing done...')

        #split train set into train-val
        x_tr, x_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

        print('training CRNN model..\n')
        model=CRNN_model(Nb_fm,k,m,DROP_OUT,num_classes)
        
	# Create loss and accuracy variables for training and validation process of all cell types
    	hist_loss_tr=[]
    	hist_loss_va=[]
    	hist_acc_tr=[]
    	hist_acc_va=[]
        # fit the model
        for iteration in range(NB_EPOCHS):
            hist=model.fit(x_tr, y_tr, validation_data=(x_va, y_va), nb_epoch=1, batch_size=BATCH_SIZE, verbose=VERBOSE)
            val_error =hist.history['val_loss']
	    # if validation loss become nan, stop training process
            if np.isnan(val_error):
                break
            hist_loss_tr.append(hist.history['loss'])
            hist_loss_va.append(hist.history['val_loss'])
            hist_acc_tr.append(hist.history['acc'])
            hist_acc_va.append(hist.history['val_acc'])
            
        # summarize performance of the model
        scores = model.evaluate(x_va,y_va, verbose=VERBOSE)
        print("Model Accuracy: %.2f%%" % (scores[1]*100))

        # save model
	model.save('my_model.h5')
         
	# prediction 
        y_pred = model.predict_classes(X_test, verbose=VERBOSE)
        y_score = model.predict_proba(X_test, verbose=VERBOSE)

	# result saving
        score = accuracy_score(y_test,y_pred)
        auc.append(roc_auc_score(y_test,y_score))
        np.savez(os.path.join(dir_name + '.npz'), AUC=auc,hist_loss_tr=hist_loss_tr,hist_loss_va=hist_loss_va,hist_acc_tr=hist_acc_tr,hist_acc_va=hist_acc_va,y_te=y_test,y_pre=y_pred,y_sco=y_score)

    np.savez(os.path.join('AUC' + '.npz'), AUC=auc)

if __name__== '__main__':
    main()
