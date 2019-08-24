from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim, matplotlib.pyplot as plt, numpy as np
import torch.nn as nn,torch,time
import Networks,DataLoader
from tqdm import tqdm 

model = Networks.Net(18)
model = torch.load('savedModels/acc_95.60011428703552_loss_26486.349609375,26487.984495162964_ep_14.pth')
gpu = True
device = torch.device('cuda')
if(torch.cuda.is_available()):model.float()
if(gpu):model.cuda()
split = 1
train_loader = DataLoader.getDataLoader(filename = "Data_RRR.h5",sampled = False,split=split)
print("Data Loaded")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),0.00001)
print(len(train_loader))
if(split<1):print(len(test_loader))
epoch = 50
maxAccuracy = 80
print("|-"*20," Learn from your mistakes ","-|"*20)
for epoch in range(epoch):
    epochStart = time.time()
    runningLoss = torch.tensor(0).to(dtype=torch.float32,device=device)
    model.train()
    for data in tqdm(train_loader):
        epochStart = time.time()
        inputs = data[0]
        labels = data[1]
        inputs,labels = Variable(inputs),Variable(labels)
        if(gpu):inputs, labels = inputs.type(torch.FloatTensor).to(device = device),labels.long().to(device=device)
        else:inputs, labels = inputs.type(torch.DoubleTensor),labels.long()
        outputs = model(inputs)
        loss = criterion(outputs,labels).type(torch.FloatTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        runningLoss += loss
    end = (time.time()-epochStart)

    print("Epoch ",epoch+1,"Loss ",runningLoss.item()/len(train_loader),"Epoch Time ",end)
    
    model.eval()
    test_loss = 0
    accuracy=0
    print(20*"-"," Let's See How Much You Learned ","-"*20)
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader):
            if(gpu):inputs, labels = inputs.type(torch.FloatTensor).to(device = device),labels.long().to(device=device)
            else:inputs, labels = inputs.type(torch.DoubleTensor),labels.long()
            out = model(inputs)
            batch_loss = criterion(out,labels)
            test_loss += batch_loss.item()
            accuracy += int(sum(out.max(1)[1]==labels))
    accuracy = 100*accuracy/(len(train_loader)*train_loader.batch_size)
    
    print("Accuracy ",accuracy, "Test Loss", test_loss/len(train_loader))
    if(accuracy > maxAccuracy):
        torch.save(model,'savedModels/acc_{}_loss_{},{}_ep_{}.pth'.format(accuracy,runningLoss,test_loss,epoch))
        maxAccuracy = accuracy
print("Saving after {} epochs".format(epoch))
torch.save(model,'savedModels/acc_{}_loss_{},{}_ep_{}_NotBalanced.pth'.format(accuracy,runningLoss,test_loss,epoch))






