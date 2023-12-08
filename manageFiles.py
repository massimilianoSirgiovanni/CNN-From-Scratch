import pickle
from colorama import Fore, Style

def saveVariableInFile(file_path, variable):
    picklefile = open(file_path, 'wb')
    pickle.dump(variable, picklefile)
    picklefile.close()

def loadVariableFromFile(file_path):
    picklefile = open(file_path, 'rb')
    variable = pickle.load(picklefile)
    picklefile.close()
    print(f"Existing variable loaded from file: {Fore.CYAN}{file_path}{Style.RESET_ALL}")
    return variable