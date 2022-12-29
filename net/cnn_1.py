import numpy as np
from layers import conv_fast
from layers import pooling
from layers import activate
from layers import fc
from layers import loss
from layers import batch_normal
from layers import dropout

class Net():
    def __init__(self):
        # (输出通道数，输入通道数，卷积核大小，卷积核大小)
        self.conv1 = conv_fast.conv((16, 1, 3, 3), stride=1, padding='VALID', bias=True, requires_grad=True)
        self.BN1 = batch_normal.BN(16, moving_decay=0.9, is_train=True)
        self.relu1 = activate.Relu()

        self.conv2 = conv_fast.conv((32, 16, 3, 3), stride=1, padding="VALID", bias=True, requires_grad=True)
        self.BN2 = batch_normal.BN(32, moving_decay=0.9, is_train=True)
        self.relu2 = activate.Relu()
        self.pooling2 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)

        self.conv3 = conv_fast.conv((64, 32, 3, 3), stride=1, padding="VALID", bias=True, requires_grad=True)
        self.BN3 = batch_normal.BN(64, moving_decay=0.9, is_train=True)
        self.relu3 = activate.Relu()

        self.conv4 = conv_fast.conv((128, 64, 3, 3), stride=1, padding="VALID", bias=True, requires_grad=True) 
        self.BN4 = batch_normal.BN(128, moving_decay=0.9, is_train=True)
        self.relu4 = activate.Relu()
        self.pooling4 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)

        self.fc5 = fc.fc(128*4*4, 512, bias=True, requires_grad=True)
        self.BN5 = batch_normal.BN(512, moving_decay=0.9, is_train=True)
        self.relu5 = activate.Relu()

        self.drop6 = dropout.Dropout(0.2,is_train=True)
        self.fc6 = fc.fc(512, 10, bias=True, requires_grad=True)

        self.softmax = loss.softmax()

    def forward(self, imgs, labels, is_train=True):
        """
        :param imgs:输入的图片：[N,C,H,W]
        :param labels:
        :return:
        """
        x = self.conv1.forward(imgs)
        x = self.BN1.forward(x, is_train)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.BN2.forward(x, is_train)
        x = self.relu2.forward(x)
        x = self.pooling2.forward(x)

        x = self.conv3.forward(x)
        x = self.BN3.forward(x,is_train)
        x = self.relu3.forward(x)

        x = self.conv4.forward(x)
        x = self.BN4.forward(x,is_train)
        x = self.relu4.forward(x)
        x = self.pooling4.forward(x)

        x = self.fc5.forward(x)
        x = self.relu5.forward(x)

        x = self.drop6.forward(x)
        x = self.fc6.forward(x)

        loss = self.softmax.calculate_loss(x, labels)
        prediction = self.softmax.prediction_func(x)
        return loss, prediction


    def backward(self, lr):
        """
        :param lr:学习率
        :return:
        """
        eta = self.softmax.gradient()

        eta = self.fc6.backward(eta, lr)
        eta = self.drop6.backward(eta)

        eta = self.relu5.backward(eta)
        eta = self.fc5.backward(eta, lr)

        eta = self.pooling4.backward(eta)
        eta = self.relu4.backward(eta)
        eta = self.BN4.backward(eta,lr)
        eta = self.conv4.backward(eta,lr)

        eta = self.relu3.backward(eta)
        eta = self.BN3.backward(eta, lr)
        eta = self.conv3.backward(eta, lr)

        eta = self.pooling2.backward(eta)
        eta = self.relu2.backward(eta)
        eta = self.BN2.backward(eta, lr)
        eta = self.conv2.backward(eta, lr)

        eta = self.relu1.backward(eta)
        eta = self.BN1.backward(eta, lr)
        eta = self.conv1.backward(eta, lr)

