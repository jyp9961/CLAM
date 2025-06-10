import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import utils
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from cifar100_labels import *
from cifar10_labels import *
from fmnist_labels import *
from mImagenet_labels import *
from collections import OrderedDict

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def Focal_Loss(probs, labels, gamma):
    selected_probs = probs[range(len(probs)), labels] + 1e-8 # avoid torch.log(0)
    # FL(p) = -log(p) * (1-p) ** \gamma 
    loss = -torch.log(selected_probs) * (1-selected_probs)**gamma

    return loss

def Pw_Loss(probs, labels, theta, gamma):
    selected_probs = probs[range(len(probs)), labels] + 1e-8 # avoid torch.log(0)
    # FL(p) = -log(p) * (\theta + (1-p) ** \gamma) 
    loss = -torch.log(selected_probs) * (theta + (1-selected_probs)**gamma)

    return loss

def get_transforms(task):
    if task in ['fmnist']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Resize(224),
            transforms.RandomResizedCrop(size=224, scale=(crop_lower_bound, 1)),
            transforms.RandomHorizontalFlip(), 
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Resize(224),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    if task in ['miniImagenet']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224,224]),
            transforms.RandomResizedCrop(size=224, scale=(crop_lower_bound, 1)),
            transforms.RandomHorizontalFlip(), 
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224,224]),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    if task in ['cifar100', 'cifar10']:
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(size=224, scale=(crop_lower_bound, 1)),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    return transform_train, transform_test

def train(task, EPOCH, crop_lower_bound, focal_loss, pw_loss, tilted_weighted_loss, GGF_loss, apstar_loss, CLAM_loss, theta, gamma, discount, min_weight, weight_frequency, l2_weight, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if focal_loss:
        exp_type = '{}_focal_loss_cropbound{}_gamma{}'.format(task, crop_lower_bound, gamma)
    elif pw_loss:
        exp_type = '{}_pw_loss_cropbound{}_theta{}_gamma{}'.format(task, crop_lower_bound, theta, gamma)
    elif tilted_weighted_loss:
        exp_type = '{}_tce_cropbound{}'.format(task, crop_lower_bound)    
    elif GGF_loss:
        exp_type = '{}_GGF_cropbound{}_discount{}_minweight{}_weightfreq{}'.format(task, crop_lower_bound, discount, min_weight, weight_frequency)
    elif apstar_loss:
        exp_type = '{}_apstar_loss_cropbound{}'.format(task, crop_lower_bound)
    elif CLAM_loss:
        exp_type = '{}_CLAM_loss_cropbound{}'.format(task, crop_lower_bound)
    else:
        # normal_loss
        exp_type = '{}_cropbound{}_l2w{}'.format(task, crop_lower_bound, l2_weight)

    print('exp_type {}'.format(exp_type))
    
    # convert to a normalized torch.FloatTensor
    transform_train, transform_test = get_transforms(task)

    print('transform_train', transform_train)
    print('transform_test', transform_test)
        
    start = time.time()
    batch_size = 128 # samples per minibatch
    print('batch_size {}'.format(batch_size))

    if task in ['cifar100', 'cifar10', 'fmnist']:
        # training dataset (random crop)
        if task == 'cifar100':
            train_data = torchvision.datasets.CIFAR100(
                root='./{}/'.format(task),
                train=True,
                download=True,
                transform = transform_train
            )
            test_data = torchvision.datasets.CIFAR100(
                root='./{}/'.format(task),
                train=False,
                download=True,
                transform = transform_test
            )

        if task == 'cifar10':
            train_data = torchvision.datasets.CIFAR10(
                root='./{}/'.format(task),
                train=True,
                download=True,
                transform = transform_train
            )
            test_data = torchvision.datasets.CIFAR10(
                root='./{}/'.format(task),
                train=False,
                download=True,
                transform = transform_test
            )
            
        if task == 'fmnist':
            train_data = torchvision.datasets.FashionMNIST(
                root='./{}/'.format(task),
                train=True,
                download=True,
                transform = transform_train
            )
            test_data = torchvision.datasets.FashionMNIST(
                root='./{}/'.format(task),
                train=False,
                download=True,
                transform = transform_test
            )    

        train_data = MyDataset(train_data)
        test_data = MyDataset(test_data)

        print("load {} datasets {:.3f}".format(task, time.time()-start))

        # indices for validation
        num_train = len(train_data)
        if task == 'cifar100':
            patience = 6
        if task in ['cifar10', 'fmnist']:
            patience = 3
        CLAM_start_epoch = 0

        train_idx = np.arange(len(train_data)) # use all data from training set to train
        
        # samplers for obtaining training batches
        train_sampler = SubsetRandomSampler(train_idx)
        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        print('len(train_loader)', len(train_loader))
        print('len(test_loader)', len(test_loader))
    
    if task in ['miniImagenet']:
        patience = 6
        CLAM_start_epoch = 50

        # training dataset (random crop)
        dataset_train = datasets.ImageFolder('./mini-imagenet/image_CDD/train', transform_train)
        train_data = MyDataset(dataset_train)
        num_train = len(train_data)
        
        train_idx = np.arange(num_train) # use all data from training set to train
        
        # samplers for obtaining training batches
        train_sampler = SubsetRandomSampler(train_idx)
        
        # test dataset
        dataset_test = datasets.ImageFolder('./mini-imagenet/image_CDD/test', transform_test)
        test_data = MyDataset(dataset_test)
        
        print("load miniImagenet datasets {:.3f}".format(time.time()-start))
        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        print('len(train_loader)', len(train_loader))
        print('len(test_loader)', len(test_loader))
    
    # initialize a resnet
    if task in ['cifar100', 'miniImagenet']:
        num_classes = 100
    if task in ['cifar10', 'fmnist']:
        num_classes = 10
    net = torchvision.models.resnet50(weights=None)
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, num_classes)
    print('number of parameters',sum(p.numel() for p in net.parameters()))
    net.to(device)
    net.apply(utils.weight_init)

    # loss function and learning rate
    criterion = nn.CrossEntropyLoss()  # use cross entropy for classification
    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    lr = 1e-1
    momentum = 0.9
    weight_decay = l2_weight
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True) # SGD as optimizer
    print('current lr {}'.format(optimizer.param_groups[0]['lr']))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, threshold=0.001, mode='max', min_lr=1e-3)
    
    print("Start Training, Resnet-50!")
    if task == 'cifar100':
        cols = cifar100_labels.copy()
    if task == 'cifar10':
        cols = cifar10_labels.copy()
    if task == 'fmnist':
        cols = fmnist_labels.copy()
    if task == 'miniImagenet':
        cols = mImagenet_labels.copy()

    tmp_cols = cols.copy()    
    tmp_cols.insert(0, 'epoch')
    tmp_cols.append('train_acc')
    train_acc_df = pd.DataFrame(columns = tmp_cols)        

    tmp_cols = cols.copy()    
    tmp_cols.insert(0, 'epoch')
    tmp_cols.append('average')
    tmp_cols.append('valid_acc')
    tmp_cols.append('train_acc')
    test_acc_df = pd.DataFrame(columns = tmp_cols)

    START = time.time()
    best_acc = 0
    log_frequency = 25

    if CLAM_loss or tilted_weighted_loss or apstar_loss:
        tmp_cols = cols.copy()
        cols.insert(0, 'epoch')
        weights_df = pd.DataFrame(columns = cols)

        weights_per_class = dict()
        for _ in range(num_classes): 
            weights_per_class[_] = 1.0

    if apstar_loss:
        K, K_min, apstar_max_loss, alpha = 1, 1, 10, 0.5
        
    num_iters = 0
    for epoch in range(EPOCH):
        start = time.time()
        if CLAM_loss or tilted_weighted_loss or apstar_loss:
            weights_df.loc[len(weights_df.index)] = np.concatenate([[epoch], [weights_per_class[_] for _ in weights_per_class]])
            weights_df.to_csv('weights_{}.csv'.format(exp_type))
        net.train()
        sum_loss, correct, total = 0.0, 0.0, 0.0
        iter_start=time.time()

        # training
        epoch_train_loader = train_loader
        for i, data in enumerate(epoch_train_loader, 0):
            # prepare the data
            num_iters += 1
            inputs, labels, indices = data
            labels_numpy, indices_numpy = labels.numpy(), indices.numpy()
            inputs, labels, indices = inputs.to(device), labels.to(device), indices.to(device)
            
            if (GGF_loss and epoch>0 and (epoch+1) % weight_frequency==0) or tilted_weighted_loss or apstar_loss or CLAM_loss:
                # losses that use weights
                outputs = net(inputs)
                loss = criterion_none(outputs, labels)
                
                batch_class_weights = np.array([weights_per_class[label] for label in labels_numpy])
                batch_class_weights = torch.tensor(batch_class_weights).reshape(1,-1).to(device)
                
                loss = batch_class_weights * loss
                    
                loss = torch.mean(loss)
                
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()

            elif focal_loss or pw_loss:
                # focal loss or pw loss
                outputs = net(inputs)

                # probs: batch_size * num_classes
                softmax = nn.Softmax(dim=1)    
                probs = softmax(outputs)
                
                if focal_loss:
                    loss = Focal_Loss(probs, labels, gamma)
                if pw_loss:
                    loss = Pw_Loss(probs, labels, theta, gamma)
                
                loss = torch.mean(loss)
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()

            else:
                # normal loss
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # loss and accuracy for each minibatch
                sum_loss += loss.item()
                total += labels.size(0)
                train_acc = correct / total
                if (i+1) % log_frequency == 0:
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %.3f ' % (epoch + 1, num_iters, sum_loss / log_frequency, 100. * train_acc, time.time()-iter_start))
                    sum_loss = 0.0

        # log accuracy in training set
        idx = 0
        with torch.no_grad():
            total_loss_per_class = dict()
            for _ in range(num_classes):
                total_loss_per_class[_] = []

            label_accuracies = np.zeros(num_classes)
            label_nums = np.zeros(num_classes)
            correct, total = 0, 0
            for i, data in enumerate(train_loader, 0):
                s = time.time()
                net.eval()
                images, labels, indices = data
                labels_numpy, indices_numpy = labels.numpy(), indices.numpy()
                images, labels, indices = images.to(device), labels.to(device), indices.to(device)
                outputs = net(images)
                
                if GGF_loss or tilted_weighted_loss or apstar_loss:
                    loss = criterion_none(outputs, labels)
                    loss_numpy = loss.detach().cpu().numpy()
                    # log train_loss_per_class
                    for i_sample in range(len(labels_numpy)):
                        total_loss_per_class[labels_numpy[i_sample]].append(loss_numpy[i_sample])
                
                # log train_acc_per_class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                correct_prediction_flag = predicted == labels
                for j in range(len(correct_prediction_flag.cpu().numpy())):
                    label_nums[labels[j].cpu().numpy()] += 1
                    if correct_prediction_flag[j]:
                        label_accuracies[labels[j].cpu().numpy()] += 1

            if CLAM_loss and epoch>=CLAM_start_epoch:
                # update weights_per_class by training accuracy
                print('label_accuracies/label_nums', label_accuracies/label_nums)
                diff_weights = np.exp(-label_accuracies / label_nums)
                diff_weights = diff_weights / np.mean(diff_weights)
                print('diff_weights', diff_weights)
                
                for class_id in range(num_classes):
                    tmp_weight = weights_per_class[class_id]
                    tmp_weight = np.clip(tmp_weight * diff_weights[class_id], 0.5, 2)
                    weights_per_class[class_id] = tmp_weight

                mean_weight = sum(weights_per_class.values()) / len(weights_per_class)
                
                # normalize the weights
                for _ in weights_per_class:
                    weights_per_class[_] /= mean_weight
                max_weight = max(weights_per_class.values())
                
                print('weights_per_class', weights_per_class)
            
            # save train acc for each class
            train_acc = correct / total
            train_acc_df.loc[len(train_acc_df.index)] = np.concatenate([[epoch], label_accuracies/label_nums, [train_acc.detach().cpu().numpy()]])
            train_acc_per_label = label_accuracies / label_nums
            print('Train Acc per label', train_acc_per_label)
            train_acc_df.to_csv('train_{}.csv'.format(exp_type))

            # update weights for each class
            if GGF_loss or tilted_weighted_loss or apstar_loss:
                for label in total_loss_per_class:
                    total_loss_per_class[label] = np.mean(total_loss_per_class[label])
                print('trainloss_per_class',total_loss_per_class)
                if GGF_loss:
                    sorted_total_loss_per_class = sorted(total_loss_per_class.items(), key=lambda x:x[1], reverse=True)
                    weights = np.array([max(discount**_,min_weight) for _ in range(len(total_loss_per_class))])
                    weights = weights / np.mean(weights)
                    for _ in range(len(total_loss_per_class)):
                        weights_per_class[sorted_total_loss_per_class[_][0]] = weights[_]
                
                if tilted_weighted_loss:
                    weights = np.exp(list(total_loss_per_class.values())) / np.sum(np.exp(list(total_loss_per_class.values())))
                    weights = weights / np.mean(weights)
                    for label in total_loss_per_class:
                        weights_per_class[label] = 0.5*weights_per_class[label] + 0.5*weights[label]

                if apstar_loss:
                    sorted_total_loss_per_class = sorted(total_loss_per_class.items(), key=lambda x:x[1], reverse=True)
                    for _ in range(len(total_loss_per_class)):
                        label = sorted_total_loss_per_class[_][0]
                        num_worst_classes = int(num_classes * 0.1)
                        if _ < num_worst_classes:
                            # increase the weights for the worst classes
                            # the value of the identity vector is divided by num_worst classes, then multiplied by num_classes (sum of weights is $n$ in our case)
                            identity_vector_value = num_classes / num_worst_classes
                            weights_per_class[label] = weights_per_class[label] * alpha + identity_vector_value / K * (1-alpha)
                        else:
                            weights_per_class[label] = weights_per_class[label] * alpha

                    # clip to avoid extreme weights 
                    # unnecessary in group fairness with limited number of groups
                    # but crucial in class fairness
                    for _ in range(num_classes):
                        weights_per_class[_] = np.clip(weights_per_class[_], 0.5, 2.0)

                    # normalize the weights 
                    for _ in range(num_classes):
                        weights_per_class[_] = weights_per_class[_] * num_classes / sum(weights_per_class.values())
                    
                    # if max loss decreases, set K to K_min
                    if sorted_total_loss_per_class[0][1] < apstar_max_loss:
                        print('worst_class_loss', sorted_total_loss_per_class[0][1])
                        apstar_max_loss = sorted_total_loss_per_class[0][1]
                        K = K_min
                    else:
                        K += 1

                print('weights_per_class',weights_per_class)

            print('[epoch:{}] | Train_Acc: {:.3f} | Time: {:.3f}'.format(epoch + 1, 100. * correct / total, time.time()-iter_start))
        
        # test the accuracy
        test_start = time.time()
        with torch.no_grad():
            label_accuracies = np.zeros(num_classes)
            label_nums = np.zeros(num_classes)
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels, indices = data
                images, labels, indices = images.to(device), labels.to(device), indices.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                correct_prediction_flag = predicted == labels                
                for j in range(len(correct_prediction_flag.cpu().numpy())):
                    label_nums[labels[j].cpu().numpy()] += 1
                    if correct_prediction_flag[j]:
                        label_accuracies[labels[j].cpu().numpy()] += 1
            test_acc = correct / total
    
        print('Test Acc per label',label_accuracies / label_nums)
        print('[epoch:{}] | Test_Acc: {:.3f} | Time: {:.3f} '.format(epoch + 1, 100. * correct / total, time.time()-iter_start))
        test_acc_df.loc[len(test_acc_df.index)] = np.concatenate([[epoch], label_accuracies/label_nums, [test_acc.detach().cpu().numpy()], [0], [train_acc.detach().cpu().numpy()]])
        test_acc_df.to_csv('{}.csv'.format(exp_type))
                    
    print("Training Finished, TotalEPOCH={} Best_Acc={:.3f} Total Time={:.3f}".format(EPOCH, best_acc,time.time()-START))
    
# training
if __name__ == "__main__":    
    parser = argparse.ArgumentParser("""CLAss-dependent Multiplicative-weights (CLAM)""")
    parser.add_argument('--task', type=str, default='cifar100')
    # task in ['cifar100', 'cifar10', 'fmnist', 'miniImagenet']
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--focal_loss', type=str, default='false') # Focal
    parser.add_argument('--pw_loss', type=str, default='false') # PW
    parser.add_argument('--tilted_weighted_loss', type=str, default='false') # TCE
    parser.add_argument('--GGF_loss', type=str, default='false') # GGF
    parser.add_argument('--apstar_loss', type=str, default='false') # APStar
    parser.add_argument('--CLAM_loss', type=str, default='false') # CLAM
    parser.add_argument('--theta', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--crop_lower_bound', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--min_weight', type=float, default=0.1)
    parser.add_argument('--weight_frequency', type=int, default=2)
    parser.add_argument('--l2_weight', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    utils.set_seed_everywhere(0)

    task = args.task
    print('task', task)
    
    focal_loss = False if args.focal_loss == 'false' else True
    pw_loss = False if args.pw_loss == 'false' else True
    tilted_weighted_loss = False if args.tilted_weighted_loss == 'false' else True
    GGF_loss = False if args.GGF_loss == 'false' else True
    apstar_loss = False if args.apstar_loss == 'false' else True
    CLAM_loss = False if args.CLAM_loss == 'false' else True
    print('focal_loss',focal_loss)
    print('pw_loss',pw_loss)
    print('tilted_weighted_loss',tilted_weighted_loss)
    print('GGF_loss', GGF_loss)
    print('apstar_loss', apstar_loss)
    print('CLAM_loss', CLAM_loss)
    
    theta = args.theta
    print('theta', theta)
    gamma = args.gamma
    print('gamma', gamma)
    crop_lower_bound = args.crop_lower_bound
    print('crop_lower_bound',crop_lower_bound)
    if task in ['cifar10']:
        discount, min_weight, weight_frequency = 0.9, 0.1, 1
        num_epochs = 100
    if task in ['fmnist']:
        discount, min_weight, weight_frequency = 0.95, 0.1, 1
        num_epochs = 50
    if task in ['cifar100']:
        discount, min_weight, weight_frequency = 0.98, 0.1, 2
        num_epochs = 150
    if task in ['miniImagenet']:
        discount, min_weight, weight_frequency = 0.95, 0.01, 2
        num_epochs = 150

    print('num_epochs', num_epochs)

    if task in ['cifar10', 'fmnist', 'miniImagenet']:
        l2_weight = 1e-3
    if task in ['cifar100']:
        l2_weight = 5e-4
    print('l2_weight', l2_weight)
    num_workers = args.num_workers
    print('num_workers', num_workers)
    train(task=task, EPOCH=num_epochs, crop_lower_bound=crop_lower_bound, focal_loss=focal_loss, pw_loss=pw_loss, tilted_weighted_loss=tilted_weighted_loss, GGF_loss=GGF_loss, apstar_loss=apstar_loss, CLAM_loss=CLAM_loss, theta=theta, gamma=gamma, discount=discount, min_weight=min_weight, weight_frequency=weight_frequency, l2_weight=l2_weight, num_workers=num_workers)
