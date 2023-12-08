import torch.optim as optim

from manageFiles import loadVariableFromFile
from manageData import seed, exists, Fore, Style
from trainingAndTestingFunctions import trainCNN
from plotting import plotModel

def loadChoosedModel(file_path):

    if not exists(file_path):
        print(
            f"{Fore.RED}ERROR: It is not possible to train without first choosing a model.\nIt is recommended to perform the Model Selection or to place a model with the following path {Fore.LIGHTRED_EX}\"{file_path}\"{Style.RESET_ALL}")
        exit(-1)
    return loadVariableFromFile(file_path)


def trainChoosenModel(tr_set, val_set, n_epochs=100, finalModelName=f"{seed}finalModel", patience=20, model=None):
    file_path = "./savedObjects/models/choosedModel/" + finalModelName
    if model is None:
        choosedModel = loadChoosedModel(file_path)
    else:
        choosedModel = model
    opt = optim.Adam(choosedModel.parameters())
    trainCNN(choosedModel, opt, tr_set, val_set, n_epochs=n_epochs, file_path=file_path, patience=patience)
    print(f"{Fore.YELLOW}Training completed for the model:{Style.RESET_ALL}")
    choosedModel = loadChoosedModel(file_path)
    print(choosedModel)
    plotModel(choosedModel)
    return choosedModel