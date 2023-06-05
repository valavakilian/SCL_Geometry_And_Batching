import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import sys
import os
from scipy import spatial
import io

import seaborn as sns
sns.set()
sns.set_context("paper")
# sns.set_style("darkgrid")
sns.set_style("whitegrid")   

from loggers import *
import argparse
import shutil

sns.set()
sns.set_context("paper")
# sns.set_style("darkgrid")
sns.set_style("whitegrid")     

# plt.rc('xtick', labelsize=18)
# plt.rc('ytick', labelsize=18)
# plt.rc('text', usetex = True)
# plt.rc('font', family='serif')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def calculate_accuracies(per_class_accuracy_list, classes, maj_classes, min_classes):

    total_balanced_accuracy_list = []
    maj_accuracy_list = []
    min_accuracy_list = []

    for per_class_accuracy in per_class_accuracy_list:
        total_balanced_accuracy_list.append(sum([per_class_accuracy[c] for c in classes]))
        maj_accuracy_list.append(sum([per_class_accuracy[c] for c in maj_classes]))
        min_accuracy_list.append(sum([per_class_accuracy[c] for c in min_classes]))
    
    return {"total": total_balanced_accuracy_list, "maj": maj_accuracy_list, "min": min_accuracy_list}


def calculate_Gram_Comparison(vector_list, G_comparitor, centering = False):
    
    G_Comparison_list = []
    d_G = G_comparitor.shape[0]
    for vector in vector_list:
        
        if centering:
            vector_center = np.mean(vector, axis=1).reshape((-1,1))
            vector = vector - vector_center
        
        if vector.shape[0] != d_G:
            vector = vector.T

        G_vector = vector @ vector.T
        G_vector_normalized = G_vector / np.linalg.norm(G_vector, ord = "fro")

        G_comparison = G_vector_normalized - G_comparitor
        G_Comparison_list.append(np.linalg.norm(G_comparison, ord = "fro"))

    return G_Comparison_list

def calculate_accuracies(args, per_class_test_accuracies, maj_classes, min_classes):

    total_test_accuracy = []
    maj_test_accuracy = []
    min_test_accuracy = []
    for epoch in range(0, len(per_class_test_accuracies)):
        total_test_accuracy.append(sum([per_class_test_accuracies[epoch][c] for c in range(0,args.K)]) / args.K)
        maj_test_accuracy.append(sum([per_class_test_accuracies[epoch][c] for c in maj_classes]) / len(maj_classes))
        min_test_accuracy.append(sum([per_class_test_accuracies[epoch][c] for c in min_classes]) / len(min_classes))
    
    return total_test_accuracy, maj_test_accuracy, min_test_accuracy


def calculate_Angles_Comparison(vector_list, K, G_comparitor, centering = False):


    normalized_GM = []
    d_G = G_comparitor.shape[0]
    for vector in vector_list:
        
        if centering:
            vector_center = np.mean(vector, axis=1).reshape((-1,1))
            vector = vector - vector_center
        
        if vector.shape[0] != d_G:
            vector = vector.T
        
        G_vector = vector @ vector.T
        GM = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                GM[i,j] = G_vector[i,j]/(np.sqrt(G_vector[i,i]*G_vector[j,j]))
        
        normalized_GM.append(GM)
        
    return [(GM - np.diag(np.diag(GM))).sum()/(K**2 - K) for GM in normalized_GM]



def calculate_Mu_Angles_List(vector_list, K, G_comparitor, centering = False):

    Angles = []

    d_G = G_comparitor.shape[0]
    vector = vector_list[-1]
        
    if centering:
        vector_center = np.mean(vector, axis=1).reshape((-1,1))
        vector = vector - vector_center
    
    if vector.shape[0] != d_G:
        vector = vector.T
    
    G_vector = vector @ vector.T
    for i in range(K):
        for j in range(K):
            if i != j:
                if i > j:
                    Angles.append( ( 1 - np.sign(G_vector[i,j]) * np.arccos(G_vector[i,j]/(np.sqrt(G_vector[i,i]*G_vector[j,j]))) / np.pi ) )
                else:
                    Angles.append( ( 1 - np.sign(G_vector[i,j]) * np.arccos(G_vector[i,j]/(np.sqrt(G_vector[i,i]*G_vector[j,j]))) / np.pi ) )
    return Angles



parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_dir', type=str, default='../data', help='path to dataset directory')
parser.add_argument('--save_dir', type=str, default='../saved_logs', help='path to experiment directory')
parser.add_argument('--img_dim', type=int, default=32)
parser.add_argument("--model", type=str, default='ResNet18', choices=['ResNet18', 'VGG', 'DenseNet', 'MLP'])
parser.add_argument("--activation", type=str, default='ReLU', choices=['ReLU', 'Linear', 'Sigmoid', "PReLU"])
parser.add_argument("--dataset", type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'FMNIST'])
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
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=350, help='number of training epochs')
parser.add_argument("--gpu", action='store_true')
parser.add_argument('--loss_type', default='SCL', type=str, choices=['CE', 'SCL', 'wSCL'], help='Imbalance loss type')
parser.add_argument('--SCL_temp', default=0.1, type=float, help='SCL temperature')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_decay_epochs', type=float, nargs='+', default = [116, 232], help='learning rate decay epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gamma', type=float, nargs='+', default = 0.0, help='Gamma value for wSCL')
parser.add_argument("--args_rand", type=int, default=1)
parser.add_argument("--debug", action="store_true")
parser.add_argument('--R_index', type=int, default=1, help='Imbalance ratio index for multirun')
parser.add_argument('--logs_path', type=str, default="./", help='Path to the logs files')
parser.add_argument('--R_range','--list', type=int, default= [1,5,10,50,100], nargs='+', help='List of imbalance values')

args = parser.parse_args()



def main(logs_path, dataset, legend = False, showTPT = False, showTPTLegend = False):

    print("_" * 100)
    print("Plotting : " + str(logs_path.split("/")[-1]))
    print("For ratios:" + str(args.R_range))
        
    if args.epochs == 500:
        log_epoch_list = [1,   3,   5,   7,   9,
                            11,  20,  30,  40,  60,
                            80, 100, 120, 140, 150, 160, 165, 170, 190, 210, 220, 240, 260, 280, 
                            300, 315, 330, 335, 350, 370, 390, 410, 425, 450, 460, 470, 480, 490,
                            495, 497, 499, 500]
    else:
        log_epoch_list = [1,   3,   5,   7,   9,
                            11,  20,  30,  40,  60,
                            80, 101, 120, 140, 160,
                            180, 201, 220, 235, 245, 250, 260,
                            275, 280, 290, 299, 305, 310, 315, 
                            320, 325, 330, 335, 340, 345, 349, 350]
    
    args.logs_path = logs_path
    args.dataset = dataset
    args.R_range = [1,10,100]

    # ------ ETF Vectors --------------------------------------------------------------------------------------------------
    G_etf = np.eye(args.K) - 1/args.K * np.ones((args.K,args.K))
    G_etf = G_etf / np.linalg.norm(G_etf, ord = "fro")

    G_OF = np.eye(args.K)
    G_OF = G_OF / np.linalg.norm(G_OF, ord = "fro")

    total_accuracy_train = {"total": {}, "maj": {}, "min": {}}
    total_accuracy_test = {"total": {}, "maj": {}, "min": {}}

    NC1_train = {}
    NC1_test = {}

    G_Mu_Centered_ETF_comparison_train = {}
    G_Mu_Centered_ETF_comparison_test = {}

    G_Mu_nonCentered_OF_comparison_train = {}
    G_Mu_nonCentered_OF_comparison_test = {}

    loss_dict_train = {}
    loss_dict_test = {}

    NCC_dict_train = {}
    NCC_dict_test = {}

    angles_train = {}
    angles_test = {}

    zero_error_epoch = {}


    R_Range = args.R_range
    colour_R_dict = {1:"gold", 2:"brown", 5:"red", 10:"green", 20:"purple", 50:"blue", 100:"black" }

    classes = [i for i in range(0, args.K)]
    maj_classes = [i for i in range(0, args.K // 2)]
    min_classes = [i for i in range(args.K//2, args.K)]


    for R in R_Range:
        train_logger_path = args.logs_path + "/R_" + str(R) + "/gamma_0.0_ver_0/logger_train.pkl"
        test_logger_path = args.logs_path + "/R_" + str(R) + "/gamma_0.0_ver_0/logger_test.pkl"
         
        if os.path.exists(train_logger_path):
            file = open(train_logger_path, 'rb')
            train_logger = CPU_Unpickler(file).load()
        else:
            print(train_logger_path)
            break
        if os.path.exists(test_logger_path):
            file = open(test_logger_path, 'rb')
            test_logger = CPU_Unpickler(file).load()
        else:
            print(test_logger_path)
            break


        angles_train[R] = calculate_Mu_Angles_List(train_logger.M, args.K, G_OF, centering = False)
        angles_test[R] = calculate_Mu_Angles_List(test_logger.M, args.K, G_OF, centering = False)

        plt.rc('xtick', labelsize=35)
        plt.rc('ytick', labelsize=35)
        plt.rc('text', usetex = True)
        plt.rc('font', family='serif')
        

        x0 = angles_train[R] 

        plt.rcParams["figure.figsize"] = [9, 3]
        plt.boxplot(x0, vert = False, patch_artist=True, showfliers = False, widths=(0.5))
        plt.xlim(0.45, 0.55)
        plt.ylim(0.5, 1.5)
        plt.xticks(ticks = np.arange(0.40, 0.60, step=0.05), labels = [str(val) for val in list(np.arange(0.40, 0.60, step=0.05))] )
        plt.yticks(ticks = [])

        if R == 1:
            plt.axvline(x=1 - np.arccos(-1/args.K) / np.pi, c="red", linewidth = 4, label = "Graf et al.")
        plt.axvline(x=1 - np.arccos(0) / np.pi, c="green", linewidth = 4, label = "OF (ours)")

        if R == 1:
            plt.legend(fontsize=30, loc = "lower left", bbox_to_anchor=(-0.04, -0.1, 0.4, 0.4), frameon=False)
        
        plt.grid(False)
        plt.savefig("Graff_Boxplot_R" + str(R) + ".pdf", bbox_inches='tight', dpi = 1200)

        plt.show()



if __name__ == '__main__':

    main("../logs/" + args.model + "_CIFAR10_step_SCL_ReLU_Augment", "CIFAR10", legend = False, showTPT = True, showTPTLegend = True)