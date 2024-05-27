import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import time
import os


from dataset import train_data, test_data
from Models.CNN import CNN, CNN_MK1 , CNN_MK2

batch_size = 60
epochs = 50
lr = 0.01
device = torch.device('cuda')

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def load_data():
    train_data_set = train_data()
    test_data_set = test_data()
    train_loader = dataloader.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_loader = dataloader.DataLoader(test_data_set, batch_size=1, shuffle=False)
    return train_loader , test_loader

def train(model,cuda=True,resume=False):
    name=model._get_name()

    train_loader,test_data = load_data()
    time0 = time.time()
    if cuda and torch.cuda.is_available():
        model = model.cuda()
        print('cuda is open')

    start_epoch = -1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.5)
    scheldule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    if resume:
        checkpoint = torch.load(os.path.join('./LP_recongnize/weights/',name+'.pth'))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheldule.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('resume from epoch:', start_epoch)
        
    for e in range(start_epoch+1,epochs):
        running_loss = 0
        model.train()
        for data in train_loader:
            if cuda and torch.cuda.is_available():
                inputs, labels = data[0].to(device), data[1].to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheldule.step()

        accuracy = val(model,test_data)
        print('epoch:', e, 'loss:', running_loss,'accuracy:' , accuracy ,'time:', (time.time() - time0)/60, 'min')
        checkpoint ={
            'net':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':e,
            'scheduler':scheldule.state_dict(),
        }

        torch.save(checkpoint, os.path.join('./LP_recongnize/weights/',name+'.pth'))
        with open(os.path.join('./LP_recongnize/weights/',name+'.txt'),'a') as f:
            f.write('epoch:'+str(e)+' loss:'+str(running_loss)+' time:'+str((time.time() - time0)/60)+'min'+ ' accuracy:'+str(accuracy)+'\n')
    
    print('Finished Training')
    torch.save(model.state_dict(), os.path.join('./LP_recongnize/weights/',name+'.pth'))

def val(model,test_loader,cuda=True):
    model.eval()
    correct = 0 
    total = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda and torch.cuda.is_available():
                images=images.to(device)
                labels=labels.to(device)
            outputs = model(images)
            outputs = outputs.view(7,35)
            labels = labels.view(7,35)
            if (outputs.argmax(dim=1) == labels.argmax(dim=1)).all():
                correct += 1
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total

def test(model,cuda=True):
    train_loader,test_loader = load_data()
    model.load_state_dict(torch.load(os.path.join('./LP_recongnize/weights/',model._get_name()+'.pth')))
    model.eval()
    correct = 0
    total = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda and torch.cuda.is_available():
                model = model.to(device)
                images=images.to(device)
                labels=labels.to(device)
            outputs = model(images)
            outputs = outputs.view(7,35)
            labels = labels.view(7,35)
            if (outputs.argmax(dim=1) == labels.argmax(dim=1)).all():
                correct += 1
            else:
                print(decode(outputs.argmax(dim=1)), decode(labels.argmax(dim=1)))
                #images = images.cpu()
                #plt.imshow(images[0].permute(1,2,0))
                #plt.show()
                pass
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total

def decode(list):
    n=''
    n+=provinces[list[0]]
    n+=alphabets[list[1]]
    for i in range(2,7):
        n+=ads[list[i]]
    return n

if __name__=='__main__':
    #train(CNN_MK2(),resume=Tr ue)
    test(CNN_MK2())
    """l1,l2=load_data()
    for i in l2:
        images, labels = i
        print(images.size(), labels.size())"""
    pass

    
