import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
# import onnx
# from onnx import numpy_helper


def mnist_fnn_6_500():
    model = nn.Sequential(
        nn.Linear(784, 500),
        nn.Sigmoid(),
        nn.Linear(500, 500),
        nn.Sigmoid(),
        nn.Linear(500, 500),
        nn.Sigmoid(),
        nn.Linear(500, 500),
        nn.Sigmoid(),
        nn.Linear(500, 500),
        nn.Sigmoid(),
        nn.Linear(500, 500),
        nn.Sigmoid(),
        nn.Linear(500, 10)
    )
    return model


def mnist_fnn_3_50():
    model = nn.Sequential(
        nn.Linear(784, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10)
    )
    return model


def mnist_fnn_5_100():
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fnn_3_100():
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10)
    )
    return model


def fashion_fnn_3_50():
    model = nn.Sequential(
        nn.Linear(784, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10)
    )
    return model


def fashion_fnn_5_100():
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10)
    )
    return model


def cifar_fnn_3_50():
    model = nn.Sequential(
        nn.Linear(3072, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10)
    )
    return model


def cifar_fnn_5_100():
    model = nn.Sequential(
        nn.Linear(3072, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10)
    )
    return model


def mnist_cnn_3l_2_3():
    model = nn.Sequential(
        nn.Linear(28*28, 2*26*26),
        nn.Sigmoid(),
        nn.Linear(2*26*26, 2*24*24),
        nn.Sigmoid(),
        nn.Linear(2*24*24, 10)
    )
    return model


def mnist_cnn_6l_5_3():
    model = nn.Sequential(
        nn.Linear(28*28, 5*26*26),
        nn.Sigmoid(),
        nn.Linear(5*26*26, 5 * 24 * 24),
        nn.Sigmoid(),
        nn.Linear(5 * 24 * 24, 5 * 22 * 22),
        nn.Sigmoid(),
        nn.Linear(5 * 22 * 22, 5 * 20 * 20),
        nn.Sigmoid(),
        nn.Linear(5 * 20 * 20, 5 * 18 * 18),
        nn.Sigmoid(),
        nn.Linear(5 * 18 * 18, 10)
    )
    return model


def fashion_cnn_4l_5_3():
    model = nn.Sequential(
        nn.Linear(28*28, 5*26*26),
        nn.Sigmoid(),
        nn.Linear(5*26*26, 5*24*24),
        nn.Sigmoid(),
        nn.Linear(5 * 24 * 24, 5 * 22 * 22),
        nn.Sigmoid(),
        nn.Linear(5*22*22, 10)
    )
    return model


def fashion_cnn_6l_5_3():
    model = nn.Sequential(
        nn.Linear(28*28, 5*26*26),
        nn.Sigmoid(),
        nn.Linear(5*26*26, 5 * 24 * 24),
        nn.Sigmoid(),
        nn.Linear(5 * 24 * 24, 5 * 22 * 22),
        nn.Sigmoid(),
        nn.Linear(5 * 22 * 22, 5 * 20 * 20),
        nn.Sigmoid(),
        nn.Linear(5 * 20 * 20, 5 * 18 * 18),
        nn.Sigmoid(),
        nn.Linear(5 * 18 * 18, 10)
    )
    return model


def cifar_cnn_3l_2_3():
    model = nn.Sequential(
        nn.Linear(3 * 32 * 32, 2 * 30 * 30),
        nn.Sigmoid(),
        nn.Linear(2 * 30 * 30, 2 * 28 * 28),
        nn.Sigmoid(),
        nn.Linear(2 * 28 * 28, 10)
    )
    return model


def cifar_cnn_6l_5_3():
    model = nn.Sequential(
        nn.Linear(3 * 32*32, 5*30*30),
        nn.Sigmoid(),
        nn.Linear(5*30*30, 5 * 28 * 28),
        nn.Sigmoid(),
        nn.Linear(5 * 28 * 28, 5 * 26 * 26),
        nn.Sigmoid(),
        nn.Linear(5 * 26 * 26, 5 * 24 * 24),
        nn.Sigmoid(),
        nn.Linear(5 * 24 * 24, 5 * 22 * 22),
        nn.Sigmoid(),
        nn.Linear(5 * 22 * 22, 10)
    )
    return model


if __name__ == '__main__':
    train = MNIST('../data/', train=True, download=False, transform=ToTensor())
    test = MNIST('../data/', train=False, download=False, transform=ToTensor())
    l = test.data.shape[0]
    test_data = torch.flatten(test.data, start_dim=1)

    model = mnist_fnn_3_50()
    model.load_state_dict(torch.load('./saved_weights/mnist_fnn_3_50_weights.pth'))
    c = 0
    for i in range(test_data.shape[0]):
        pred = model(test_data[i] / 255)
        if torch.argmax(pred).item() == test.targets[i].item():
            c += 1
    print(c)
    quit()

    model = mnist_fnn_6_500()
    # model.load_state_dict(torch.load('model_weights.pth'))
    standard_listed = list(model.state_dict().keys())
    stated = torch.load('model_weights.pth')
    listed = list(stated.keys())

    for i in range(16):
        stated.pop(listed[i])
    new_listed = list(stated.keys())

    for i in range(len(standard_listed)):
        stated.update({standard_listed[i]: stated[new_listed[i]]})
        stated.pop(new_listed[i])
    model.load_state_dict(stated)

    # onx_model = onnx.load('ffnnSIGMOID__PGDK_w_0.3_6_500.onnx')
    # for ele in onx_model.graph.node:
    #     # print(numpy_helper.to_array(ele).shape)
    #     if ele.name == 'Flatten_4':
    #         print(ele)

    c = 0
    for i in range(100):
        data = ((test_data[i] / 255) - 0.1307) / 0.3081
        pred = model(data)
        if torch.argmax(pred).item() == test.targets[i]:
            c += 1
    print(c)
    paras = model.parameters()
    for e in paras:
        print(e.shape)
    # for e in model.children():
    #     print(e.state_dict().keys())
    # torch.save(model.state_dict(), './eran/mnist_6_500_PGD0.3.pth')
