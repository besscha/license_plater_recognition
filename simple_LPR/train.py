from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import time

from dataset import train_data, test_data
from Model.CNN_first import CNN_first

batch_size = 30
epochs = 30
lr = 0.001
device = torch.device('cuda')

def load_data():
    train_data_set = train_data()
    test_data_set = test_data()
    train_loader = dataloader.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_loader = dataloader.DataLoader(test_data_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

def train(model,cuda=True):
    train_loader,test_loader = load_data()
    time0 = time.time()
    if cuda and torch.cuda.is_available():
        model = model.cuda()
        print('cuda is open')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.5)
    scheldule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for e in range(epochs):
        running_loss = 0
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
        print('epoch:', e, 'loss:', running_loss, 'time:', (time.time() - time0)/60, 'min')
        torch.save(model.state_dict(), 'simple_LPR/weight/cnn_first.pth')

def test(model,cuda=True):
    train_loader,test_loader = load_data()
    model.load_state_dict(torch.load('simple_LPR/weight/cnn_first.pth'))
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
                print(outputs.argmax(dim=1), labels.argmax(dim=1))
                images = images.cpu()
                plt.imshow(images[0].permute(1,2,0))
                plt.show()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


if __name__== '__main__':
    model=CNN_first()
    #train(model)
    test(model)
    '''
    train_loader,test_loader = load_data()
    i=iter(train_loader)
    img,lab = next(i)
    plt.imshow(img[0].permute(1,2,0))
    plt.show()
    print(img.shape,lab.shape)
    lab=lab[0].view(7,35)
    print(lab.argmax(dim=1))
    '''