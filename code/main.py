
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
import pickle

import os
import pickle
import shutil

from generate_cifar import IMBALANCECIFAR10
from generate_mnist import IMBALANCEMNIST
import matplotlib.pyplot as plt

import argparse

from models import *

from criterion import *

from loggers import *

from utils import *

import gc
gc.collect()



parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_dir', type=str, default='../data', help='path to dataset directory')
parser.add_argument('--save_dir', type=str, default='../saved_logs', help='path to experiment directory')
parser.add_argument('--img_dim', type=int, default=32)
parser.add_argument("--model", type=str, default='ResNet18', choices=['ResNet18', 'VGG', 'DenseNet', 'MLP'])
parser.add_argument("--activation", type=str, default='ReLU', choices=['ReLU', 'Linear', 'Sigmoid', "PReLU"])
parser.add_argument("--dataset", type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'FMNIST'])
parser.add_argument("--repeated_examples", action='store_true')
parser.add_argument("--K", type=int, default=10)
parser.add_argument('--imb_type', type=str, default="step", choices=['exp', 'step'], help='Imbalance Type')
parser.add_argument('--R', type=int, default=10, help='Imbalance ratio')
parser.add_argument('--rho', type=float, default=0.5, help='Step imbalance cutoff')
parser.add_argument("--n_maj", type=int, default=0)
parser.add_argument("--augmentation", action='store_true')
parser.add_argument('--noAugmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument("--perBatchAugmentation", action='store_true')
parser.add_argument('--noPerBatchAugmentation', dest='perBatchAugmentation', action='store_false')
parser.set_defaults(perBatchAugmentation=True)
parser.add_argument("--versions", type=int, default=1)
parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=350, help='number of training epochs')
parser.add_argument("--gpu", action='store_true')
parser.add_argument('--loss_type', default='SCL', type=str, choices=['CE', 'SCL'], help='Imbalance loss type')
parser.add_argument('--SCL_temp', default=0.1, type=float, help='SCL temperature')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_decay_epochs', type=float, nargs='+', default = [116, 232], help='learning rate decay epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument("--args_rand", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


# Hook for this experiment's features
class features:
    pass
def hook(self, input, output):
    features.value = input[0].clone()


def main():

    G_OF = np.eye(args.K)
    G_OF = G_OF / np.linalg.norm(G_OF, ord = "fro")

    if args.epochs == 500:
        log_epoch_list = [1,   3,   5,   7,   9,
                        11,  20,  30,  40,  60,
                        80, 100, 120, 140, 150, 160, 165, 170, 190, 210, 220, 240, 260, 280, 
                        300, 315, 330, 335, 350, 370, 390, 410, 425, 450, 460, 470, 480, 490,
                        495, 497, 499, 500]
    elif args.epochs == 350:
        log_epoch_list = [1,   3,   5,   7,   9,
                        11,  20,  30,  40,  60,
                        80, 101, 120, 140, 160,
                        180, 201, 220, 235, 245, 250, 260,
                        275, 280, 290, 299, 305, 310, 315, 
                        320, 325, 330, 335, 340, 345, 349, 350]


    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    torch.set_default_dtype(torch.float32)

    classes = [c for c in range(0, args.K)]
    maj_classes = [c for c in range(0, int(args.K * args.rho))]
    min_classes = [c for c in range(0, args.K) if c not in maj_classes]
    delta_list = [args.R if c in maj_classes else 1 for c in range(0, args.K)]

    if args.n_maj == 0:
        args.n_maj = 5000

    if args.imb_type == "step":
        n_c_train = [args.n_maj if c in maj_classes else int(args.n_maj // args.R) for c in range(0, args.K)]
    else:
        n_c_train = [int(args.n_maj * ((1/args.R) ** (c / (args.K - 1.0)))) for c in range(0,args.K)]
    

    augmentation = "Augment" if args.augmentation is True else "noAugment"
    if args.repeated_examples:
        general_save_dir = args.save_dir + "/" + '_'.join([args.model, args.dataset, args.imb_type, args.loss_type, args.activation, augmentation]) 
        general_save_dir = general_save_dir + "_repeatedExamples"
        general_save_dir = general_save_dir + "/R_" + str(args.R) + "/" 
    else:
        general_save_dir = args.save_dir + "/" + '_'.join([args.model, args.dataset, args.imb_type, args.loss_type, args.activation, augmentation]) + "/R_" + str(args.R) + "/" 
    general_save_dir_model = args.save_dir + "_model/" + '_'.join([args.model, args.dataset, args.imb_type, args.loss_type, args.activation, augmentation]) + "/R_" + str(args.R) + "/" 
    if not os.path.exists(general_save_dir):
        os.makedirs(general_save_dir, exist_ok=True)
    if not os.path.exists(general_save_dir_model):
        os.makedirs(general_save_dir_model, exist_ok=True)

    f = open(general_save_dir + "print_logs.txt", "w")
    f.write("Create File!\n")
    f.flush()
    f.write(str(args))
    f.flush()
    f.write("Starting SCL Experiments!\n")
    f.flush()


    # ------- Imbalanced dataset --------------------------------------------------------------------------------------------------
    if args.dataset == 'CIFAR10':
        input_ch        = 3
        im_size = 32
        padded_im_size = 32

        test_transforms_list = [ transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        train_transforms_list = test_transforms_list.copy() 
        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(test_transforms_list)

        train_dataset = IMBALANCECIFAR10(args.data_dir, imb_type=args.imb_type, imb_factor= 1/args.R,
                                        rand_number=args.args_rand, train=True, download=True,
                                        transform=transform_train, n_c_train_target=n_c_train, classes=classes, n_maj = args.n_maj)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)

        if augmentation == "Augment":
            train_transforms_augment_list = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            train_transforms_augment_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
            train_transforms_augment_list.insert(1, transforms.RandomCrop(im_size, padding=4))
            train_transforms_augment_list.insert(2, transforms.RandomHorizontalFlip())
            transform_augment_train = transforms.Compose(train_transforms_augment_list)
        else:
            transform_augment_train = None

    elif args.dataset == 'MNIST':
        input_ch        = 1
        im_size = 28
        padded_im_size = 32

        test_transforms_list = [ transforms.ToTensor(),
                                 transforms.Normalize(0.1307, 0.3081)]
        train_transforms_list = test_transforms_list.copy() 
        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(test_transforms_list)

        train_dataset = IMBALANCEMNIST(args.data_dir, args.imb_type, imb_factor= 1/args.R,
                                      rand_number=args.args_rand, train=True, download=True,
                                      transform=transform_train, n_c_train_target=n_c_train,
                                      classes=classes, n_maj = args.n_maj)
        train_dataset.data = torch.tensor(train_dataset.data)

        val_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform_test)

        if augmentation == "Augment":
            train_transforms_augment_list = [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
            train_transforms_augment_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
            train_transforms_augment_list.insert(1, transforms.RandomCrop(im_size, padding=4))
            train_transforms_augment_list.insert(2, transforms.RandomHorizontalFlip())
            transform_augment_train = transforms.Compose(train_transforms_augment_list)
        else:
            transform_augment_train = None
    
    elif args.dataset == 'FMNIST':
        input_ch        = 1
        im_size = 28
        padded_im_size = 32

        transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        train_dataset = IMBALANCEMNIST(args.data_dir, args.imb_type, imb_factor= 1/args.R,
                                      rand_number=args.args_rand, train=True, download=True,
                                      transform=transform, n_c_train_target=n_c_train,
                                      classes=classes, n_maj = args.n_maj)
        train_dataset.data = torch.tensor(train_dataset.data)

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)


    # Adding the repeated examples, take on sample from each class
    # Could add more examples per class. 
    if args.repeated_examples:
        f.write("Running in normal mode" + "\n")
        f.flush()
        repeated_examples = []
        for c in range(0, args.K):
            for data in train_dataset:
                if int(data[1]) == c:
                    repeated_examples.append(data)
                    break
        repeated_examples_dataloder = torch.utils.data.DataLoader(repeated_examples, batch_size = args.K, shuffle=False, pin_memory=True, sampler=None)
    else:
        f.write("Running in normal mode" + "\n")
        f.flush()
        repeated_examples_dataloder = None
    

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, sampler=None)

    analysis_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader( val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # ------- Main Training -------------------------------------------------------------------------------------------------------
    for version in range(0, args.versions):
        print("-" * 30)
        print("Performing experiment for " + str('_'.join(['gamma', str(args.gamma), "ver", str(version)])))

        exp_save_path = general_save_dir + '_'.join(['gamma', str(args.gamma), "ver", str(version)]) + "/"
        exp_complete_flag = exp_save_path + "exp_complete.txt"

        exp_save_path_model = general_save_dir_model + '_'.join(['gamma', str(args.gamma), "ver", str(version)]) + "/"

        if not os.path.exists(exp_save_path):
            os.makedirs(exp_save_path, exist_ok=True)
            os.makedirs(exp_save_path_model, exist_ok=True)
        if not os.path.exists(exp_complete_flag):
            shutil.rmtree(exp_save_path)
            os.makedirs(exp_save_path, exist_ok=True)
        else:
            print("Skipping this experiments since flag is set. Please remove flag to rerun this experiment.")
            continue

        # ------- Model -------------------------------------------------------------------------------------------------------

        if args.model == "ResNet18":

            if args.loss_type == "SCL":
                model = ResNet18(args.K, args.loss_type , input_ch)
                classifier = model.core_model.fc
                classifier_hook = None
                model = model.to(device)

            if args.loss_type == "CE":
                model = models.resnet18(pretrained=False, num_classes=args.K)
                model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
                model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                classifier = model.fc
                classifier.register_forward_hook(hook)
                classifier_hook = None
                model = model.to(device)

        if args.model == "DenseNet":
            model = DenseNet40(args.K, args.loss_type, input_ch)
            classifier = model.core_model.classifier
            classifier_hook = None
            model = model.to(device)
        
        
        save_path = exp_save_path_model + "model_init.pth"
        torch.save(model.state_dict(), save_path)

        # ------- Loss ---------------------------------------------------------------------------------------------------------
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss()
            criterion_analysis = nn.CrossEntropyLoss()
        if args.loss_type == "SCL":
            criterion = SupConLoss(temperature = 0.1)
            criterion_analysis = SupConLoss(temperature = 0.1)

        # ------- Optimizer ----------------------------------------------------------------------------------------------------
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=args.lr_decay_epochs,
                                                    gamma=args.lr_decay)
        
        # ------- Data Loggers --------------------------------------------------------------------------------------------------
        logger_train = logger()
        logger_test = logger()


        OF_convergence = []

        # ------- Train ---------------------------------------------------------------------------------------------------------
        cur_epochs = []
        for epoch in range(1, args.epochs + 1):
            print("print epoch")
            torch.cuda.empty_cache()

            train(model, criterion, args, device, train_loader, optimizer, epoch, repeated_examples_dataloder, transform_augment_train = transform_augment_train)
            lr_scheduler.step()
            
            if epoch in log_epoch_list:
                cur_epochs.append(epoch)
                Mu_train = analysis(logger_train, model, criterion_analysis, args, device, analysis_loader, classifier, classifier_hook, epoch)
                analysis(logger_test, model, criterion_analysis, args, device, test_loader, classifier, classifier_hook, epoch, Mu_for_NCC = Mu_train)

                save_logger(logger_train, exp_save_path, "logger_train")
                save_logger(logger_test, exp_save_path, "logger_test")

                save_path = exp_save_path_model + "model_epoch_" + str(epoch) + ".pth"
                torch.save(model.state_dict(), save_path)
                
                G_vector = logger_train.M[-1].T @ logger_train.M[-1]
                G_vector_normalized = G_vector / np.linalg.norm(G_vector, ord = "fro")

                G_comparison = G_vector_normalized - G_OF
                OF_comparison = np.linalg.norm(G_comparison, ord = "fro")

                f.write("------------------------------\n")
                f.write("Epoch " + str(epoch) + " -> \n")
                f.write("reg_loss: " + str(logger_train.loss[-1]) + " -> \n")
                f.write("NCC train accuracy: " + str(logger_train.NCC_acc_perclass[-1]) + " -> \n")
                f.write("NCC test accuracy: " + str(logger_test.NCC_acc_perclass[-1]) + " -> \n")
                f.write("OF_comparison: " + str(OF_comparison) + " -> \n")
                f.flush()

                OF_convergence.append(OF_comparison)

                print("OF_comparison: " + str(OF_comparison) + " -> \n")

        os.makedirs(exp_complete_flag, exist_ok=True)
    
    f.close()


if __name__ == '__main__':
    main()