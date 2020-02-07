# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:03:38 2020

@author: EI-LAP-7241
"""



import torch
import torch.nn as nn
from sklearn import metrics

class CustomLossFunc():
    
    def loss_func(self,y_actual,y_predicted):
    
        delta=torch.ones_like(y_actual)
        delta=torch.where(y_actual>=0,delta,y_actual)
        y_predicted=y_predicted*delta
        y_predicted=y_predicted[y_predicted>=0]
        y_actual=y_actual[y_actual>=0]
        l=nn.BCELoss()
        loss=l(y_predicted,y_actual)
        p=y_predicted.detach().numpy()
        y=y_actual.detach().numpy()
        
        accuracy=metrics.accuracy_score(y,p.round())
        
        
             
        return loss,accuracy,len(y)