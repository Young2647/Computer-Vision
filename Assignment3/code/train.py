import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from torchvision import transforms
from net import Net
import dataload

def train(train_loader, net, optimizer, epoch, device) :
    criterion = nn.L1Loss()

    net.train()
    running_loss = 0
    total_num = 0
    for i, sample in enumerate(train_loader) :
        image = sample['image'].to(device)
        depth = sample['depth'].to(device)

        optimizer.zero_grad() #zero the parameter gradients

        #forward + backward + optimize
        output = net(image)
        loss = criterion(output,depth)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_num += batch_size

    #print statistics
    print("epoch : {}, loss : {}".format(epoch, running_loss/float(total_num)))

if __name__ == "__main__" :
    lr_ = 0.001 # learning rate
    weight_decay_ = 1e-4 
    batch_size = 8
    epoch_num = 20 #total epoch num
    train_net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_net.to(device)

    optimizer = torch.optim.Adam(train_net.parameters(), lr=lr_, weight_decay=weight_decay_)
    train_loader = dataload.load_data('./nyuv2/train/', batch_size)

    for epoch in range (epoch_num) :
        if (epoch % 5 == 0) :
            lr_ = lr_ * 0.1 #reduce learning rate to 10% every 5 epochs
            optimizer = torch.optim.Adam(train_net.parameters(), lr=lr_, weight_decay=weight_decay_)
        train(train_loader, train_net, optimizer, epoch, device)
    
    PATH = './train_net.pth'
    torch.save(train_net.state_dict(),PATH)