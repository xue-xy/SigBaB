import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor
import numpy as np
from model.models import *
from model_verification import VModel
from bab_verification import robustness
from time import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='choose a saved model')
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--device', default='cuda:0', help='cpu or gpu')
parser.add_argument('--eps', default=0.015, help='radius')
parser.add_argument('--branch', default='improvement', choices=['max', 'integral', 'improvement'], help='branching strategy')
parser.add_argument('--split', default='convex', choices=['zero', 'half', 'inflection', 'convex'], help='neuron split method')
parser.add_argument('--bab', default=True, type=bool, help='whether or not to use bab')
parser.add_argument('--batch_size', default=400, type=int, help='batch size')
parser.add_argument('--tlimit', default=300, help='time limit for each property')
args = parser.parse_args()

net_name = 'cifar_fnn_3_50'

if net_name == 'mnist_fnn_6_500':
    net = mnist_fnn_6_500()
    net.load_state_dict(torch.load('./model/eran/mnist_6_500_PGD0.3.pth'))
    model = VModel(net, mean=0.1307, std=0.3081, device=args.device)
elif net_name == 'mnist_fnn_3_50':
    net = mnist_fnn_3_50()
    net.load_state_dict(torch.load('./model/saved_weights/mnist_fnn_3_50_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'mnist_fnn_5_100':
    net = mnist_fnn_5_100()
    net.load_state_dict(torch.load('./model/saved_weights/mnist_fnn_5_100_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'mnist_fnn_3_100':
    net = mnist_fnn_3_100()
    net.load_state_dict(torch.load('./model/saved_weights/mnist_fnn_3_100_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'fashion_fnn_3_50':
    net = fashion_fnn_3_50()
    net.load_state_dict(torch.load('./model/saved_weights/fashion_fnn_3_50_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'fashion_fnn_5_100':
    net = fashion_fnn_5_100()
    net.load_state_dict(torch.load('./model/saved_weights/fashion_fnn_5_100_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
# model = VModel(net, mean=0.1307, std=0.3081, device=args.device)
elif net_name == 'cifar_fnn_3_50':
    net = cifar_fnn_3_50()
    net.load_state_dict(torch.load('./model/saved_weights/cifar_fnn_3_50_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'cifar_fnn_5_100':
    net = cifar_fnn_5_100()
    net.load_state_dict(torch.load('./model/saved_weights/cifar_fnn_5_100_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'mnist_cnn_3l_2_3':
    net = mnist_cnn_3l_2_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/mnist_cnn_3l_2_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'mnist_cnn_6l_5_3':
    net = mnist_cnn_6l_5_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/mnist_cnn_6l_5_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'fashion_cnn_4l_5_3':
    net = fashion_cnn_4l_5_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/fashion_cnn_4l_5_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'fashion_cnn_6l_5_3':
    net = fashion_cnn_6l_5_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/fashion_cnn_6l_5_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'cifar_cnn_3l_2_3':
    net = cifar_cnn_3l_2_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/cifar_cnn_3l_2_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)
elif net_name == 'cifar_cnn_6l_5_3':
    net = cifar_cnn_6l_5_3()
    net.load_state_dict(torch.load('./model/saved_weights/cnn/cifar_cnn_6l_5_3_weights.pth'))
    model = VModel(net, mean=0, std=1, device=args.device)

if net_name[0] == 'm':
    test_data = MNIST('./data/', train=False, download=True, transform=ToTensor())
    data = torch.flatten(test_data.data / 255, start_dim=1)
    labels = test_data.targets
elif net_name[0] == 'f':
    test_data = FashionMNIST('./data/', train=False, download=True, transform=ToTensor())
    data = torch.flatten(test_data.data / 255, start_dim=1)
    labels = test_data.targets
elif net_name[0] == 'c' and 'fnn' in net_name:
    test_data = CIFAR10('./data', train=False, download=True, transform=ToTensor())
    data = torch.flatten(torch.tensor(test_data.data) / 255, start_dim=1)
    labels = torch.tensor(test_data.targets)
elif net_name[0] == 'c' and 'cnn' in net_name:
    test_data = CIFAR10('./data', train=False, download=True, transform=ToTensor())
    data = torch.flatten(torch.tensor(np.transpose(test_data.data, [0, 3, 1, 2])) / 255, start_dim=1)
    labels = torch.tensor(test_data.targets)

bab = args.bab
eps = args.eps
check_num = 1000
image_proved = [2] * check_num
prop_true = 0
prop_false = 0
prop_und = 0
time_record = []

t1 = 0
f1 = 0
u1 = 0
print(net_name)
print(args)
corret_count = 0

for i in tqdm(range(0, 100)):
    pred = torch.argmax(model.model((data[i].to(args.device) - model.mean) / model.std)).item()
    if pred != labels[i]:
        continue
    else:
        corret_count += 1

    start_time = time()

    l_ans, p_true, p_false, und = robustness(model, data[i], labels[i], args.eps, args)
    end_time = time()
    model.reset()
    print(l_ans, p_true, p_false, und)

    # args.branch = 'integral'
    # args.bab = False
    # l_ans1, p_true1, p_false1, und1 = robustness(model, data[i], labels[i], eps, args)
    # model.reset()
    #
    # print(i)
    # print(l_ans, p_true, p_false, und)
    # print(l_ans1, p_true1, p_false1, und1)
    # if l_ans1 == l_ans and p_true1 == p_true and p_false1 == p_false and und1 == und:
    #     print(True)
    # else:
    #     print(False)
    # if p_false1 > p_false:
    #     print('Wrong')
    # print('-'*40)
    # t1 += p_true1
    # f1 += p_false1
    # u1 += und1

    image_proved[i] = l_ans
    prop_true += p_true
    prop_false += p_false
    prop_und += und
    time_record.append(end_time - start_time)
    # print(i)

if args.bab:
    if net_name == 'mnist_fnn_3_50':
        np.save('./time/m350/b{}_t{}_{}_sig_100_{}{}_3-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch, args.split), np.array(time_record))
    elif net_name == 'mnist_fnn_5_100':
        np.save('./time/m5100/b{}_t{}_{}_sig_100_{}{}_4-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    if net_name == 'mnist_fnn_3_100':
        np.save('./time/m3100/b{}_t{}_{}_sig_100_{}{}_4-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch, args.split), np.array(time_record))
    if net_name == 'mnist_fnn_6_500':
        np.save('./time/m6500/b{}_t{}_{}_sig_100_{}{}_4-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch, args.split), np.array(time_record))
    elif net_name == 'fashion_fnn_3_50':
        np.save('./time/f350/b{}_t{}_{}_sig_100_{}{}_4-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                 args.split), np.array(time_record))
    elif net_name == 'fashion_fnn_5_100':
        np.save('./time/f5100/b{}_t{}_{}_sig_100_{}{}_4-4.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'cifar_fnn_3_50':
        np.save('./time/c350/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'cifar_fnn_5_100':
        np.save('./time/c5100/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'mnist_cnn_3l_2_3':
        np.save('./time/cnn/m3l23/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'mnist_cnn_6l_5_3':
        np.save('./time/cnn/m6l53/b{}_t{}_{}_sig_100_{}{}_2-2.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'fashion_cnn_4l_5_3':
        np.save('./time/cnn/f4l53/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'fashion_cnn_6l_5_3':
        np.save('./time/cnn/f6l53/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'cifar_cnn_3l_2_3':
        np.save('./time/cnn/c3l23/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))
    elif net_name == 'fashion_cnn_6l_5_3':
        np.save('./time/cnn/c6l53/b{}_t{}_{}_sig_100_{}{}.npy'.format(args.batch_size, args.tlimit, eps, args.branch,
                                                                     args.split), np.array(time_record))


print('correct count:', corret_count)
print('image level: -------------')
print('proved true: {}, proved false: {}, undecidable: {}'.format(image_proved.count(1), image_proved.count(-1), image_proved.count(0)))
print('property level: ------------------')
print('proved true: {}, proved false: {}, undecidable: {}'.format(prop_true, prop_false, prop_und))

# # print()
# print('property level: ------------------')
# print('baseline')
# print('proved true: {}, proved false: {}, undecidable: {}'.format(t1, f1, u1))

