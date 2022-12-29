import cupy as np
from functools import reduce
from layers.parameter import parameter

class fc():
    def __init__(self, input_num, output_num, bias=True, requires_grad=True):
        """
        :param input_num:输入神经元个数
        :param output_num: 输出神经元的个数
        """
        self.input_num = input_num          # 输入神经元个数
        self.output_num = output_num        # 输出神经元个数
        self.requires_grad = requires_grad
        self.weight = parameter(np.random.randn(self.input_num, self.output_num) * (2/self.input_num**0.5))
        if bias:
            self.bias = parameter(np.random.randn(self.output_num))
        else:
            self.bias = None


    def forward(self, input):
        """
        :param input: 输入的feature map 形状：[N,C,H,W]或[N,C*H*W]
        :return:
        """
        self.input_shape = input.shape    # 记录输入数据的形状
        if input.ndim > 2:
            N, C, H, W = input.shape
            self.x = input.reshape((N, -1))
        elif input.ndim == 2:
            self.x = input
        else:
            print("fc.forward的输入数据维度存在问题")
        result = np.dot(self.x, self.weight.data)
        if self.bias is not None:
            result = result + self.bias.data
        return result


    def backward(self, eta, lr):
        """
        :param eta:由上一层传入的梯度 形状：[N,output_num]
        :param lr:学习率
        :return: self.weight.grad 回传到上一层的梯度
        """
        N, _ = eta.shape
        # 计算传到下一层的梯度
        next_eta = np.dot(eta, self.weight.data.T)
        self.weight.grad = np.reshape(next_eta, self.input_shape)

        # 计算本层W,b的梯度。这里有两种不同的实现方式，但是本质是一样的，不信你仔细的推一下。
        # 实现一
        x = self.x.repeat(self.output_num, axis=0).reshape((N, self.output_num, -1))
        self.W_grad = x * eta.reshape((N, -1, 1))
        self.W_grad = np.sum(self.W_grad, axis=0) / N
        self.b_grad = np.sum(eta, axis=0) / N

        # 权重更新
        self.weight.data -= lr * self.W_grad.T
        self.bias.data -= lr * self.b_grad

        return self.weight.grad



