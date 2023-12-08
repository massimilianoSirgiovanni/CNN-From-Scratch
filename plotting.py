import matplotlib.pyplot as pp

def plotVal_err(val_error, selectModel=-1):
    pp.plot(val_error)
    if selectModel > -1:
        pp.vlines(x=selectModel, ymin=0, ymax=1, colors='tab:gray', linestyles='dashdot')
    pp.title("Error on validation set through the epochs")
    pp.xlabel("Epochs")
    pp.ylabel("Training Error")
    pp.draw()
    pp.show()


def plotTr_err(tr_error, selectModel=-1):
    pp.plot(tr_error)
    if selectModel > -1:
        pp.vlines(x=selectModel, ymin=0, ymax=1, colors='tab:gray', linestyles='dashdot')
    pp.title("Training error through the epochs")
    pp.xlabel("Epochs")
    pp.ylabel("Training Error")
    pp.draw()
    pp.show()


def plotTrAndVal_err(tr_error, val_error, selectModel=1):
    pp.plot(tr_error, 'g-', label="training")
    pp.plot(val_error, 'r--', label="validation")
    if selectModel > -1:
        pp.vlines(x=selectModel, ymin=0, ymax=1, colors='tab:gray', linestyles='dashdot')
    pp.title("Training and Validation errors through the epochs")
    pp.xlabel("Epochs")
    pp.ylabel("Errors")
    pp.legend(loc="upper left")
    pp.draw()
    pp.show()

def plotModel(model):
    tr_error = model.tr_error
    tr_error.extend(model.earlyStoppingTrErr)
    val_error = model.val_error
    val_error.extend(model.earlyStoppingValErr)
    plotTrAndVal_err(tr_error, val_error, selectModel=model.bestEpoch)