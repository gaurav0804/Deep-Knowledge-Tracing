# -*- coding: utf-8 -*-

import numpy as np
import itertools
import torch




class ReadProcessData():
    def __init__(self,train_path,test_path,max_len,max_ques):
        self.train_path=train_path
        self.test_path=test_path
        self.maxlen=max_len
        self.maxques=max_ques
    
    def readProcessData(self):
        
        X_train=[]
        y_train=[]
        X_test=[]
        y_test=[]


        with open(self.train_path, 'r') as train:
                    for length, ques, ans in itertools.zip_longest(*[train] * 3):
                        length = int(length.strip().strip(','))
                        ques = [int(q)+1 for q in ques.strip().strip(',').split(',')]
                        ans = [int(a) for a in ans.strip().strip(',').split(',')]
                        ans_array=np.zeros(shape=(self.maxlen,self.maxques-1))
                        ans_array=ans_array-1
                        slices=(length//self.maxlen)+1
                        for slice_ in range(slices):
                            ques_=list(ques[slice_*self.maxlen:(slice_+1)*self.maxlen])
                            ans_=list(ans[slice_*self.maxlen:(slice_+1)*self.maxlen])
                            for i in range(len(ques_)):
                                if ans_[i]==1:
                                    ans_array[i,ques_[i]-1]=1
                                elif ans_[i]==0:
                                    ans_array[i,ques_[i]-1]=0
                                                                        
                            filler=[0]*(self.maxlen-len(ques_))
                            ques_=ques_+filler
                            ans_=[a+1 for a in ans_]
                            ans_=ans_+filler
                            ans_=[a-1 for a in ans_]
                            ans_=np.array(ans_)
                            ans_=ans_*125
                            ques_=np.array(ques_)
                            ques_=ques_+ans_
                            ques_=list(ques_)
                            ques_=[max(q,0) for q in ques_]
                            
                            X_train.append(ques_)
                            y_train.append(ans_array)
                            
        with open(self.test_path, 'r') as train:
                    for length, ques, ans in itertools.zip_longest(*[train] * 3):
                        length = int(length.strip().strip(','))
                        ques = [int(q)+1 for q in ques.strip().strip(',').split(',')]
                        ans = [int(a) for a in ans.strip().strip(',').split(',')]
                        ans_array=np.zeros(shape=(self.maxlen,self.maxques-1))
                        ans_array=ans_array-1
                        slices=(length//self.maxlen)+1
                        for slice_ in range(slices):
                            ques_=list(ques[slice_*self.maxlen:(slice_+1)*self.maxlen])
                            ans_=list(ans[slice_*self.maxlen:(slice_+1)*self.maxlen])
                            for i in range(len(ques_)):
                                if ans_[i]==1:
                                    ans_array[i,ques_[i]-1]=1
                                elif ans_[i]==0:
                                    ans_array[i,ques_[i]-1]=0
                            
                            filler=[0]*(self.maxlen-len(ques_))
                            ques_=ques_+filler
                            ans_=[a+1 for a in ans_]
                            ans_=ans_+filler
                            ans_=[a-1 for a in ans_]
                            ans_=np.array(ans_)
                            ans_=ans_*125
                            ques_=np.array(ques_)
                            ques_=ques_+ans_
                            ques_=list(ques_)
                            ques_=[max(q,0) for q in ques_]
                            
                            X_test.append(ques_)
                            y_test.append(ans_array)
        
        X_train=torch.LongTensor(np.array(X_train))
        X_test=torch.LongTensor(np.array(X_test))
        y_train=torch.FloatTensor(np.array(y_train))
        y_test=torch.FloatTensor(np.array(y_test))
        
        return X_train,X_test,y_train,y_test

