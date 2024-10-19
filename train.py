import numpy
import torch, torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import os
from RestNet18 import RestNet18
from tqdm import tqdm


################################################################################################################
# 定义训练函数
def target2class(target):
    target_table={0: '飞机 (airplane)',
                  1: '汽车 (automobile)',
                  2: '鸟 (bird)',
                  3: '猫 (cat)',
                  4: '鹿 (deer)',
                  5: '狗 (dog)',
                  6: '青蛙 (frog)',
                  7: '马 (horse)',
                  8: '船 (ship)',
                  9: '卡车 (truck)'}
    return target_table[target]



################################################################################################################
# 定义测试函数
def test(model, opt_name, log, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    acc = 0.0
    sum = 0.0
    loss_sum = 0
    file = open(log, 'a')
    msg = 'The result of {0} with {1} is'.format(model.name, opt_name)
    print(msg, end='')
    file.write(msg)
    accs_class = torch.zeros(10).to(device)
    for batch, (data, target) in enumerate(testloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        acc_index = (torch.argmax(output, dim=1) == target)
        acc += torch.sum(acc_index).item()
        for i in range(10):
            accs_class[i] += torch.sum(target[acc_index] == i)
        sum += len(target)
        loss_sum += loss.item()
    msg = ' test acc: {0}%, loss: {1}'.format(100 * acc / sum, loss_sum / (batch + 1))
    print(msg)
    file.write(msg+'\n')
    for i in range(10):
        msg = '\t test acc for {0} is {1}%'.format(target2class(i),accs_class[i]/10)
        print(msg)
        file.write(msg+'\n')
    file.close()

def train(model, optimizer, opt_name, num_epoch, log, log_test, criterion):
    # 如果有gpu就使用gpu，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    file = open(log, 'a')
    file.write('-' * 50)
    msg = 'Start training {0} with {1}'.format(model.name, opt_name)
    print(msg)
    file.write('\n' + msg + '\n')
    for epoch in range(1, num_epoch + 1):
        acc = 0.0
        sum = 0.0
        loss_sum = 0
        print('epoch {} is starting...'.format(epoch))
        for batch, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # top1_accs
            acc += torch.sum(torch.argmax(output, dim=1) == target).item()
            sum += len(target)
            loss_sum += loss.item()
        msg = '\tepoch {0}: train acc: {1}%, loss: {2}'.format(epoch, 100 * acc / sum, loss_sum / (batch + 1))
        print(msg)
        file.write(msg + '\n')

        if epoch % 5==0:
            file.close()
            saveDir = 'model_weights_epoch{}'.format(epoch)
            torch.save(model.state_dict(), saveDir)
            test(model, opt_name, log_train, criterion)
            file = open(log, 'a')
    file.close()


def cal_mean_std(CIFAR_PATH = "./data", batch_size=100):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar10_cal = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=False, transform=transform)
    cal_train = torch.utils.data.DataLoader(cifar10_cal, batch_size=batch_size, shuffle=False, num_workers=0)

    # calculate mean
    mean = torch.zeros(3)
    num=0
    for batch, (data, target) in enumerate(cal_train):
        tmp = torch.mean(torch.mean(data,dim=0).reshape(3,-1),dim=-1)
        mean += tmp
        num += 1
    mean = mean/num
    # print(mean[0].item())
    # print(mean[1].item())
    # print(mean[2].item())

    # calculate std
    std = torch.zeros(3)
    num=0
    for batch, (data, target) in enumerate(cal_train):
        tmp = torch.transpose(data, 0, 1).reshape(3,-1) - mean.reshape((3,1))
        std += torch.sum(tmp * tmp, dim=-1)
        num += data.shape[0]*data.shape[2]*data.shape[3]
    std = torch.sqrt(std/num)
    # print(std[0].item())
    # print(std[1].item())
    # print(std[2].item())
    return mean, std


if __name__ == '__main__':
    ################################################################################################################
    # 定义参数
    num_epoch = 50
    batch_size = 100
    weight_decay = 1e-3
    learning_rate = 1e-4
    log_test = 'log/log_test.txt'
    log_train = 'log/log_train.txt'

    ################################################################################################################
    # 加载数据集
    CIFAR_PATH = "./data"

    # mean, std = cal_mean_std()
    mean = [0.49139991, 0.48215881, 0.44653094]
    std = [0.24703227, 0.24348517, 0.26158783]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar10_training = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=False,
                                                    transform=transform_train)
    cifar10_testing = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=False,
                                                   transform=transform_test)

    trainloader = torch.utils.data.DataLoader(cifar10_training, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=batch_size, shuffle=False, num_workers=0)


    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # ################################################################################################################
    model = RestNet18()

    print('#' * 80)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt_name = 'Adam'
    train(model, optimizer, opt_name, num_epoch, log_train, log_test, criterion)

