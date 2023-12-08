import torch as th
from colorama import Fore, Style
from os import makedirs

from modelSelection import modelSelection
from testing import testChoosedModel
from training import trainChoosenModel
from manageData import getDatasets, seed, exists

th.manual_seed(seed)
modelSelectionEpochs = 100
trainingEpochs = 500
modelSelectionPatience = 5
trainingPatience = 30
finalModelName = f"{seed}final{trainingPatience}Patience"
modelSelectionSeeds = [seed]
modelSelectionKernels = [2, 3, 4, 5]
modelSelectionStrides = [1, 2, 3, 4, 5]

if not exists("./savedObjects"):
    makedirs("./savedObjects")

if not exists("./savedObjects/datasets"):
    makedirs("./savedObjects/datasets")

if not exists("./savedObjects/models"):
    makedirs("./savedObjects/models")

if not exists("./savedObjects/models/choosedModel"):
    makedirs("./savedObjects/models/choosedModel")

print("--------------------Starting Execution---------------------")
print(f"{Fore.YELLOW}Seed = {seed}{Style.RESET_ALL}\n")
print(f"Enter the corresponding number to choose what you want to execute:\n{Fore.LIGHTGREEN_EX}1. Download and preprocessing on data\n2. Model Selection\n3. Train the choosen model\n4. Testing the choosen model\n5. Complete Execution{Style.RESET_ALL}")

value = input("Enter your choice here: ")

if value not in ["1", "2", "3", "4", "5"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
else:
    if value == "5":
        tr_set, val_set, test_set = getDatasets()
        modelSelection(tr_set, val_set, kernels=modelSelectionKernels, strides=modelSelectionStrides, seeds=modelSelectionSeeds, epochs=modelSelectionEpochs, finalModelName=finalModelName, patience=modelSelectionPatience)
        trainChoosenModel(tr_set, val_set, n_epochs=trainingEpochs, finalModelName=finalModelName, patience=trainingPatience)
        testChoosedModel(test_set, finalModelName=finalModelName)
    else:
        tr_set, val_set, test_set = (0, 0, 0)
        if value == "1":
            tr_set, val_set, test_set = getDatasets()
            print(f"The dataset has been downloaded, divided into {Fore.BLUE}Training{Style.RESET_ALL}, {Fore.BLUE}Validation{Style.RESET_ALL} and {Fore.BLUE}Test sets{Style.RESET_ALL}.\nThe three portions of the dataset have been saved in the folder {Fore.BLUE}\"./savedObjects/datasets\"{Style.RESET_ALL}")
            forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "2"
            else:
                exit(0)
        if value == "2":
            if tr_set == 0 or val_set == 0:
                tr_set, val_set, test_set = getDatasets()
            modelSelection(tr_set, val_set, kernels=modelSelectionKernels, strides=modelSelectionStrides, seeds=modelSelectionSeeds, epochs=modelSelectionEpochs, finalModelName=finalModelName, patience=modelSelectionPatience)
            forward = input(f"{Fore.MAGENTA}Do you want to continue running the Training for the Choosed Model? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "3"
            else:
                exit(0)
        if value == "3":
            if tr_set == 0 or val_set == 0:
                tr_set, val_set, test_set = getDatasets()
            trainChoosenModel(tr_set, val_set, n_epochs=trainingEpochs, finalModelName=finalModelName, patience=trainingPatience)
            forward = input(
                f"{Fore.MAGENTA}Do you want to continue running the Testing for the Choosed Model? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "4"
            else:
                exit(0)
        if value == "4":
            if test_set == 0:
                test_set = getDatasets()[2]
            testChoosedModel(test_set, finalModelName=finalModelName)

