import sys

sys.path.append("/home/caijunhong/ml/multi-networks/minpy_cnn")
import cupy as cp
import numpy as np
import glob
import struct
import time
from net import cnn_1
from tools import normalization
import matplotlib.pyplot as plt
import os

path = r'../result/lr1_batch64_conv_epoch60.csv'

train_images = cp.load("../../archive/train_data.npy")
train_labels =  cp.load("../../archive/train_label.npy")
test_images = cp.load("../../archive/test_data.npy")

def checkpoints_all(net):
    net.conv1.checkpoints(1)
    net.conv2.checkpoints(2)
    net.conv3.checkpoints(3)
    net.conv3.checkpoints(4)

    net.BN1.checkpoints(1)
    net.BN2.checkpoints(2)
    net.BN3.checkpoints(3)
    net.BN4.checkpoints(4)

    net.fc5.checkpoints(5)
    net.fc6.checkpoints(6)
    net.fc7.checkpoints(7)
    net.fc8.checkpoints(8)
    print("***** 参数保存成功！ *****")


batch_size = 64  # 训练时的batch size
test_batch = 50  # 测试时的batch size
epoch = 60
learning_rate = 0.1

ax = []  # 保存训练过程中x轴的数据（训练次数）
ay_loss = []  # 保存训练过程中y轴的数据（loss）
ay_acc = []
testx = [] # 保存测试过程中x轴的数据（训练次数）
testy_acc = []  # 保存测试过程中y轴的数据（loss）
iterations_num = 0 # 记录训练的迭代次数

net = cnn_1.Net()
print(path)
# 训练阶段
for E in range(epoch):
    batch_loss = 0
    batch_acc = 0

    epoch_loss = 0
    epoch_acc = 0

    if E % 15 == 0 and E != 0:
        learning_rate = learning_rate / 2.71828

    for i in range(train_images.shape[0] // batch_size):
        img = train_images[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1, 28, 28)
        img = normalization.normalization(img)
        label = train_labels[i*batch_size:(i+1)*batch_size]
        # 前向传播
        loss, prediction = net.forward(img, label, is_train=True)   # 训练阶段

        epoch_loss += loss
        batch_loss += loss
        for j in range(prediction.shape[0]):
            if cp.argmax(prediction[j]) == label[j]:
                epoch_acc += 1
                batch_acc += 1

        # 反向传播
        net.backward(learning_rate)

        if (i+1)%50 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S") +
                  "   epoch:%5d , batch:%5d , avg_batch_acc:%.4f , avg_batch_loss:%.4f , lr:%f "
                  % (E+1, i+1, batch_acc/(batch_size*50), batch_loss/(batch_size*50), learning_rate))
            batch_loss = 0
            batch_acc = 0

    # 一个epoch保存一次参数
    # checkpoints_all(net)

    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          "    **********epoch:%5d , avg_epoch_acc:%.4f , avg_epoch_loss:%.4f *************"
          % (E+1, epoch_acc/train_images.shape[0], epoch_loss/train_images.shape[0]))

# 在test set上进行预测
pre = []
for k in range(test_images.shape[0] // test_batch):
    img = test_images[k*test_batch:(k+1)*test_batch].reshape(test_batch, 1 ,28, 28)
    img = normalization.normalization(img)
    _, prediction = net.forward(img, label, is_train=False)   # 测试阶段
    prediction = cp.argmax(prediction,axis=1)
    pre.append(prediction)
pre = cp.array(pre).reshape(20000)
y_out = cp.zeros([20000,2])
y_out[range(20000),0] = range(20000)
y_out[range(20000),1] = pre
cp.savetxt(path,y_out,delimiter=',',fmt="%d")

