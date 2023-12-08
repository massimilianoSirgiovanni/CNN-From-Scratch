from trainingAndTestingFunctions import testCNN
from training import loadChoosedModel, Fore, Style, seed

def testChoosedModel(test_set, model=None, finalModelName=f"{seed}finalModel"):
    file_path = "./savedObjects/models/choosedModel/" + finalModelName
    if model is None:
        choosedModel = loadChoosedModel(file_path)
    else:
        choosedModel = model
    test_accuracy, _ = testCNN(choosedModel.getPredictor(), test_set)
    print(f"{Fore.MAGENTA}Testing completed{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Test Accuracy: {Fore.GREEN}{test_accuracy}{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}Test Error: {1-test_accuracy}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Training Error: {choosedModel.tr_error[-1]}{Style.RESET_ALL}")
    return test_accuracy