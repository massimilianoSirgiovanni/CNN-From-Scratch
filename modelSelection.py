import torch.nn as nn
import torch.optim as optim
import torch as th

import models as mo
from manageData import classes, exists, Fore, Style, seed
from trainingAndTestingFunctions import trainCNN
from manageFiles import loadVariableFromFile, saveVariableInFile
from plotting import plotModel


def computePoolingOutputForSquaredKernels(inputDim, kernelConv, strideConv, kernelPool, stridePool, dimPaddingK=-1):
    dimPaddingC = int(mo.ceil((kernelConv - 1) / 2))
    dim_new = int(mo.floor((inputDim + 2 * dimPaddingC - kernelConv) / strideConv + 1))
    if dimPaddingK==-1:
        dimPaddingK = int(mo.ceil((kernelPool - 1) / 2))
    dim_pooling = int(mo.floor((dim_new + 2 * dimPaddingK - kernelPool) / stridePool + 1))
    return dim_pooling



def modelSelection(tr_set, val_set, kernels, strides, seeds=[seed], finalModelName=f"{seed}finalModel", epochs=100, patience=5, overwriteFinal=False):

    print(F"\nCONVOLUTIONAL FILTER - MODEL SELECTION ON {Fore.GREEN}KERNEL_SIZE AND STRIDE{Style.RESET_ALL}")
    val_accuracy_list = th.zeros(size=(len(seeds), len(kernels), len(strides)))
    for i in range(0, len(seeds)):
        th.manual_seed(seeds[i])
        for s in range(0, len(strides)):
            for k in range(0, len(kernels)):
                kernel_size = kernels[k]
                stride = strides[s]
                if kernel_size >= stride:

                    print(f"\n{Fore.YELLOW}Seed={seeds[i]}{Style.RESET_ALL}: Starting Model Selection for model with Convolutional Filter: {Fore.BLUE}kernel_size={Fore.GREEN}({kernel_size}, {kernel_size}){Style.RESET_ALL} and {Fore.BLUE}stride={Fore.GREEN}{stride} {Style.RESET_ALL}")
                    outputNL1 = computePoolingOutputForSquaredKernels(32, kernel_size, stride, 2, 2)
                    outputNL2 = computePoolingOutputForSquaredKernels(outputNL1, 2, 1, 2, 2)
                    outputNL3 = computePoolingOutputForSquaredKernels(outputNL2, 2, 1, 2, 2)
                    linear = outputNL3*outputNL3*30
                    nl1 = mo.Sequential(
                        mo.ConvolutionalFilter(kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding='same', numChannels=3, numFilters=10),
                        nn.ReLU(), mo.Pooling(kernel_size=(2, 2), stride=(2, 2), poolingFunction='max'))
                    nl2 = mo.Sequential(
                        mo.ConvolutionalFilter(kernel_size=(2, 2), stride=(1, 1), padding='same', numChannels=10, numFilters=20),
                        nn.ReLU(), mo.Pooling(kernel_size=(2, 2), stride=(2, 2), poolingFunction='max'))
                    nl3 = mo.Sequential(
                        mo.ConvolutionalFilter(kernel_size=(2, 2), stride=(1, 1), padding='same', numChannels=20, numFilters=30),
                        nn.ReLU(), mo.Pooling(kernel_size=(2, 2), stride=(2, 2), poolingFunction='max'))
                    output_layer = mo.Sequential(nn.Flatten(), mo.Linear(linear, len(classes), bias=True))
                    myCNN = mo.Sequential(nl1, nl2, nl3, output_layer, long_print=1)
                    file_path = f'./savedObjects/models/{seeds[i]}NL1-CF{nl1.models[0].kernel_size}-{nl1.models[0].stride}P{nl1.models[2].kernel_size}-{nl1.models[2].stride}NL2-CF{nl2.models[0].kernel_size}-{nl2.models[0].stride}P{nl2.models[2].kernel_size}-{nl2.models[2].stride}NL3-CF{nl3.models[0].kernel_size}-{nl3.models[0].stride}P{nl3.models[2].kernel_size}-{nl3.models[2].stride}'
                    if exists(file_path):
                        myCNN = loadVariableFromFile(file_path)

                    opt = optim.Adam(myCNN.parameters())
                    trainCNN(myCNN, opt, tr_set, val_set, n_epochs=epochs, file_path=file_path, patience=patience)
                    print(f"This model has been trained for:{Fore.GREEN} {myCNN.trainedEpochs} {Style.RESET_ALL}epochs")
                    print(f"{Fore.YELLOW}Seed={seeds[i]}{Style.RESET_ALL}: Executed model with Convolutional Filter: {Fore.BLUE}kernel_size={Fore.GREEN}({kernel_size}, {kernel_size}){Style.RESET_ALL} and {Fore.BLUE}stride={Fore.GREEN}{stride} {Style.RESET_ALL} ended up with {Fore.BLUE}Val Accuracy = {Fore.GREEN}{myCNN.VALAccuracy}{Style.RESET_ALL}")
                    val_accuracy_list[i, k, s] = myCNN.VALAccuracy

        print(f"\nVALIDATION ACCURACY TABLE {Fore.YELLOW}Seed={seeds[i]}{Style.RESET_ALL}:")
        print(val_accuracy_list[i])
        print()

    tot_val_accuracy_list = th.sum(val_accuracy_list, dim=0)/len(seeds)
    print(f"VALIDATION ACCURACY TABLE {Fore.MAGENTA}ON ALL SEEDS:")
    print(tot_val_accuracy_list)
    print(f"{Style.RESET_ALL}")
    kernel_index, stride_index = divmod(tot_val_accuracy_list.argmax().item(), tot_val_accuracy_list.shape[1])
    #kernel_index, stride_index = 2, 3
    print(f"Best Validation accuracy is: {Fore.GREEN}{tot_val_accuracy_list[kernel_index][stride_index]}{Style.RESET_ALL}")
    bestSeed = th.argmax(val_accuracy_list[:, kernel_index, stride_index])
    finalModelKernelSize = kernels[kernel_index]
    finalModelStride = strides[stride_index]
    choosenModel = loadVariableFromFile(f'./savedObjects/models/{seeds[bestSeed]}NL1-CF({finalModelKernelSize}, {finalModelKernelSize})-({finalModelStride}, {finalModelStride})P{nl1.models[2].kernel_size}-{nl1.models[2].stride}NL2-CF{nl2.models[0].kernel_size}-{nl2.models[0].stride}P{nl2.models[2].kernel_size}-{nl2.models[2].stride}NL3-CF{nl3.models[0].kernel_size}-{nl3.models[0].stride}P{nl3.models[2].kernel_size}-{nl3.models[2].stride}')
    if not exists(f"./savedObjects/models/choosedModel/{finalModelName}") or overwriteFinal == True:
        saveVariableInFile(f"./savedObjects/models/choosedModel/{finalModelName}", choosenModel)

    print("The Model Selection has been completed.\nThe Model Chosen through this process is the following:")
    print(choosenModel)
    print()
    plotModel(choosenModel)
    print()
    th.manual_seed(seed)
    return choosenModel