# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 16:40
# @Author  : jhys
# @FileName: train.py

import torch
import torchvision
import torch.nn as nn
from LeNetTest.model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=True, num_workers=0)

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                         shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
#
# def imshow(img):  # 展示测试集图片和标签
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_label))
net = LeNet()
net.to(device)
print(net)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 1000 == 999:
            with torch.no_grad():
                outputs = net(test_image.to(device))  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]

                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy) )  # 打印epoch，step，loss，accuracy
                print('%f s' % (time.perf_counter() - time_start))        # 打印耗时
                running_loss = 0


print('finish training!')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)