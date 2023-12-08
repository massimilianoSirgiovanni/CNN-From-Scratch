from manageFiles import saveVariableInFile, loadVariableFromFile
import numpy as np
import torch as th
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

def testCNN(my_model, test_dataset, desc='Test: '):
    test_set_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    all_pred = []
    tot_correct = 0
    with th.no_grad():
        for (batch_imgages, batch_pred) in tqdm(test_set_loader, desc=desc):
            my_pred = th.argmax(my_model(batch_imgages), -1)
            all_pred.append(my_pred)
            tot_correct += th.sum(my_pred == batch_pred).item()
    return tot_correct/len(test_dataset), th.concatenate(all_pred)


def trainCNN(my_model, opt, tr_dataset, val_dataset, n_epochs, file_path, patience=10):
    indexPatient = patience - my_model.earlyStoppingIt
    trerr_list = my_model.earlyStoppingTrErr
    valerr_list = my_model.earlyStoppingValErr
    bestValidation = my_model.VALAccuracy

    if indexPatient <= 0:
        return -1


    tr_set_loader = data.DataLoader(tr_dataset, batch_size=1000, shuffle=True)
    CE_loss = nn.CrossEntropyLoss()

    for e in range(my_model.getTrainedEpochs(), n_epochs):
        all_batch_loss = []
        for (batch_imgages, batch_pred) in tqdm(tr_set_loader, desc='Training: '):
            opt.zero_grad()
            my_pred = my_model(batch_imgages)
            loss_val = CE_loss(my_pred, batch_pred)
            loss_val.backward()
            opt.step()

            all_batch_loss.append(loss_val.detach().item())
        tot_epoch_loss = np.array(all_batch_loss).mean()
        val_accuracy, _ = testCNN(my_model, val_dataset, desc='Validation Error: ')
        tr_accuracy, _ = testCNN(my_model, tr_dataset, desc='Training Error: ')
        trerr_list.append(1 - tr_accuracy)
        valerr_list.append(1 - val_accuracy)

        print(f'Epoch {e}\t|\tTR Loss: {tot_epoch_loss:0.5f}\t|\tVAL Accuracy:{val_accuracy:0.2f}')
        my_model.trainedEpochs = my_model.trainedEpochs + 1
        
        if val_accuracy > bestValidation:
            my_model.foundBestModel(valerr_list, trerr_list, val_accuracy)
            saveVariableInFile(file_path, my_model)
            bestValidation = val_accuracy
            indexPatient = patience
            trerr_list = []
            valerr_list = []

        else:
            indexPatient = indexPatient - 1
            my_model.epochTrainedNoBestModel(trerr_list, valerr_list, earlyStoppingIt=(patience - indexPatient))
            saveVariableInFile(file_path, my_model)
            print(f"EARLY STOPPING: Patience Epochs remained {indexPatient}")
            if indexPatient == 0:
                return -1
    else:
        return -1

    return 0
