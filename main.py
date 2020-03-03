# -*- coding: utf-8 -*-



import torch
import torch.optim as optim
import numpy as np
import LSTMModel
import ReadProcessData
import CustomLossFunc
import Variables

train_path=Variables.train_path
test_path=Variables.test_path
max_len=Variables.maxlen
max_ques=Variables.maxques

datareader=ReadProcessData.ReadProcessData(train_path,test_path,max_len,max_ques)
X_train,X_test,y_train,y_test=datareader.readProcessData()

trainset=[]
for i in range(len(X_train)):
    trainset.append([X_train[i], y_train[i]])
testset=[]
for i in range(len(X_test)):
    testset.append([X_test[i], y_test[i]])


trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=100)


input_dim=Variables.input_dim
embed_dim=Variables.embed_dim
hidden_dim=Variables.hidden_dim
layer_dim=Variables.layer_dim
output_dim=Variables.output_dim
lstm=LSTMModel.LSTM(input_dim,embed_dim,hidden_dim,layer_dim,output_dim)

optimizer = optim.Adam(lstm.parameters(), lr=0.01)
CL=CustomLossFunc.CustomLossFunc()


train_losses=[]
test_losses=[]
for epoch in range(100):
    train_accuracy=[]
    train_batch=[]
    test_accuracy=[]
    test_batch=[]
    train_epoch_loss=0
    test_epoch_loss=0
    for batch, (data,target) in enumerate(trainloader):
        lstm.train()        
        pred=lstm(data)
        loss,accuracy,leny=CL.loss_func(target,pred)
        train_epoch_loss+=loss.item()
        train_accuracy.append(accuracy)

        train_batch.append(leny)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for batch, (data,target) in enumerate(testloader):
        lstm.eval()
        test_pred=lstm(X_test)
        test_loss,accuracy,leny =CL.loss_func(y_test,test_pred)
        test_epoch_loss+=test_loss.item()
        test_accuracy.append(accuracy)
        test_batch.append(leny)
        
    train_epoch_loss=train_epoch_loss/len(trainloader)
    test_epoch_loss=test_epoch_loss/len(testloader)
    train_losses.append(train_epoch_loss)
    
    test_losses.append(test_epoch_loss)

    accuracy_train=np.sum(np.array(train_accuracy)*np.array(train_batch))/np.sum(np.array(train_batch))
    accuracy_test=np.sum(np.array(test_accuracy)*np.array(test_batch))/np.sum(np.array(test_batch))
    print(f'epoch: {epoch}, train loss : {train_epoch_loss}, train accuracy : {accuracy_train}, test loss : {test_epoch_loss}, test accuracy : {accuracy_test}', )
    
    




