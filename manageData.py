import torchvision
import torch as th
import matplotlib.pyplot as plt
from os.path import exists
import torch.utils.data as data
from colorama import Fore, Style

from manageFiles import saveVariableInFile, loadVariableFromFile



seed = 12446

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def getDatasets(choosenSeed=seed):
    if not exists(f'./savedObjects/datasets/{choosenSeed}tr_dataset') or not exists(f'./savedObjects/datasets/{choosenSeed}val_dataset') or not exists(f'./savedObjects/datasets/{choosenSeed}test_dataset'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        # Images Normalization
        X = th.tensor(trainset.data / 255)
        X1 = th.index_select(X, 3, th.tensor(0)).squeeze(3).unsqueeze(1)       # Reshape Input
        X2 = th.index_select(X, 3, th.tensor(1)).squeeze(3).unsqueeze(1)
        X3 = th.index_select(X, 3, th.tensor(2)).squeeze(3).unsqueeze(1)
        X = th.cat((X1, X2, X3), 1)
        Y = th.tensor(trainset.targets)
        print(f"Input shape = {X.shape}")

        reduce = 0          # Debug Variable usefull to reduce input dimension
        N = X.shape[0] - reduce
        val_split = 0.2
        te_split = 0.2
        idx_rand = th.randperm(N)
        N_val = int(N*val_split)
        N_te = int(N*te_split)
        N_tr = N - N_val - N_te
        idx_tr = idx_rand[:N_tr]
        idx_test = idx_rand[N_tr:N_tr + N_te]
        idx_val = idx_rand[N_tr + N_te:]
        X_tr, Y_tr = X[idx_tr], Y[idx_tr]
        X_val, Y_val = X[idx_val], Y[idx_val]
        X_test, Y_test = X[idx_test], Y[idx_test]

        tr_dataset = data.TensorDataset(X_tr, Y_tr)
        saveVariableInFile(f'./savedObjects/datasets/{choosenSeed}tr_dataset', tr_dataset)
        print(f"{Fore.CYAN}Training Set Size: {Fore.GREEN}{len(tr_dataset)}{Style.RESET_ALL}")
        val_dataset = data.TensorDataset(X_val, Y_val)
        saveVariableInFile(f'./savedObjects/datasets/{choosenSeed}val_dataset', val_dataset)
        print(f"{Fore.CYAN}Validation Set Size: {Fore.GREEN}{len(val_dataset)}{Style.RESET_ALL}")
        test_dataset = data.TensorDataset(X_test, Y_test)
        saveVariableInFile(f'./savedObjects/datasets/{choosenSeed}test_dataset', test_dataset)
        print(f"{Fore.CYAN}Test Set Size: {Fore.GREEN}{len(test_dataset)}{Style.RESET_ALL}")
    else:
        tr_dataset = loadVariableFromFile(f'./savedObjects/datasets/{choosenSeed}tr_dataset')
        print(f"{Fore.CYAN}Training Set Size: {Fore.GREEN}{len(tr_dataset)}{Style.RESET_ALL}")
        val_dataset = loadVariableFromFile(f'./savedObjects/datasets/{choosenSeed}val_dataset')
        print(f"{Fore.CYAN}Validation Set Size: {Fore.GREEN}{len(val_dataset)}{Style.RESET_ALL}")
        test_dataset = loadVariableFromFile(f'./savedObjects/datasets/{choosenSeed}test_dataset')
        print(f"{Fore.CYAN}Test Set Size: {Fore.GREEN}{len(test_dataset)}{Style.RESET_ALL}")
    return tr_dataset, val_dataset, test_dataset


def displayImage(X, Y, ids):
    f, ax_list = plt.subplots(1, len(ids), figsize=(32, 32))
    ax_list = ax_list.reshape(-1)
    for i in range(len(ax_list)):
        plt.sca(ax_list[i])
        plt.imshow(X[ids[i]], origin='upper')
        plt.title(f'Class: {Y[ids[i]]} ({classes[Y[ids[i]]]})')
    plt.show()

