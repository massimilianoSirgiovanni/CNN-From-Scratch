import math
import sys
from abc import abstractmethod
from numpy import ceil, floor
import torch as th
import torch.nn as nn
from colorama import Fore, Style



class Model():
    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Sequential(Model):


    def __init__(self, *args, long_print=0):
        self.models = [*args]
        self.long_print = long_print
        self.bestEpoch = -1
        self.trainedEpochs = 0
        self.VALAccuracy = 0
        self.val_error = []
        self.tr_error = []
        self.earlyStoppingIt = 0
        self.earlyStoppingValErr = []
        self.earlyStoppingTrErr = []
        self.bestModel = self

    def __call__(self, dataset):
        output = dataset
        for model in self.models:
            output = model(output)
        return output

    def foundBestModel(self, val_error, tr_error, ValAccuracy):
        self.VALAccuracy = ValAccuracy
        self.bestEpoch = self.trainedEpochs - 1
        self.bestModel = self
        self.val_error.extend(val_error)
        self.val_error = self.val_error
        self.tr_error.extend(tr_error)
        self.tr_error = self.tr_error
        self.earlyStoppingIt = 0
        self.earlyStoppingValErr = []
        self.earlyStoppingTrErr = []


    def epochTrainedNoBestModel(self, trerr_list, valerr_list, earlyStoppingIt):
        self.earlyStoppingIt = earlyStoppingIt
        self.earlyStoppingValErr = valerr_list
        self.earlyStoppingTrErr = trerr_list

    def getTrainedEpochs(self):
        return self.trainedEpochs

    def parameters(self):
        parameters = []
        for model in self.models:
            tmp = model.parameters()
            if type(tmp) != 'list':
                tmp = list(tmp)
            parameters.extend(tmp)
        return parameters

    def getPredictor(self):
        return self.bestModel




    def __str__(self):
        output_string = f"{Fore.GREEN}----Sequential Layer----\n"
        if self.long_print == 1:
            output_string = output_string + f"Trained for {self.trainedEpochs} epochs\nBest Epoch: {self.bestEpoch}\n"
        output_string = output_string + f"Contain: [\n"
        for model in self.models:
            output_string = output_string + Fore.GREEN + str(model) + Style.RESET_ALL + "\n"
        output_string = output_string + f"{Fore.GREEN}]"
        if self.long_print == 1:
            output_string = output_string + f"\nValidation Accuracy: {self.VALAccuracy}"
        output_string = output_string + f"{Style.RESET_ALL}"
        return output_string


class ConvolutionalFilter(nn.Module):

    def __init__(self, kernel_size, stride=(1, 1), padding='zero', dimPadding='auto', numChannels=3, numFilters=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.numChannels = numChannels
        self.numFilters = numFilters
        self.filters = nn.Parameter(th.randn(numFilters, numChannels, self.kernel_size[0], self.kernel_size[1]))
        self.namePadding = padding
        if dimPadding != 'auto' and type(dimPadding) != tuple:
            print(f"{Fore.RED}ConvolutionalFilter: Invalid padding dimension: it must be a tuple or use the string \'auto\'{Style.RESET_ALL}")
            exit(-1)
        self.dimPadding = dimPadding
        if padding == 'zero':
            self.paddingFunction = self.zeroPadding
        elif padding == 'same':
            self.paddingFunction = self.samePadding
        elif padding == 'casual':
            self.paddingFunction = self.casualPadding
        elif padding == 'valid':
            self.paddingFunction = 0
        else:
            print(
                f"{Fore.MAGENTA}Warning: The string \'{padding}\' does not match any padding implemented, so no padding will be added.\nThe options are:\n > \'valid\'\n > \'zero\'\n > \'same\'\n > \'casual\'{Style.RESET_ALL}")
            self.paddingFunction = 0

    def parameters(self):
        return [self.filters]

    def zeroPadding(self, image, dimPadding, dim):
        return image

    def casualPadding(self, image, dimPadding, dim):
        rowUp = th.rand(dim[0], dim[1], dimPadding[0],  dim[3])
        rowDown = th.rand(dim[0], dim[1], dimPadding[0], dim[3])

        image[:, :, 0:dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]] = rowUp
        image[:, :, image.shape[2] - dimPadding[0]:image.shape[2], dimPadding[1]:dim[3] + dimPadding[1]] = rowDown

        columnLeft = th.rand(dim[0], dim[1], image.shape[2], dimPadding[1])
        columnRight = th.rand(dim[0], dim[1], image.shape[2], dimPadding[1])
        image[:, :, 0:image.shape[2], 0:dimPadding[1]] = columnLeft
        image[:, :, 0:image.shape[2], image.shape[3] - dimPadding[1]:image.shape[3]] = columnRight
        return image

    def samePadding(self, image, dimPadding, dim):

        rowUp = image[:, :, dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]]
        rowDown = image[:, :, dim[2], dimPadding[1]:dim[3] + dimPadding[1]]
        for i in range(0, dimPadding[0]):
            image[:, :, i, dimPadding[1]:dim[3] + dimPadding[1]] = rowUp
            image[:, :, image.shape[2] - (i + 1), dimPadding[1]:dim[3] + dimPadding[1]] = rowDown

        columnLeft = image[:, :, 0:image.shape[2], dimPadding[1]]
        columnRight = image[:, :, 0:image.shape[2], dim[3]]
        for i in range(0, dimPadding[1]):
            image[:, :, 0:image.shape[2], i] = columnLeft
            image[:, :, 0:image.shape[2], image.shape[3] - (i + 1)] = columnRight

        return image

    def addPadding(self, image):
        dim = list(image.shape)
        if self.dimPadding == 'auto':
            dimPadding = list(map(lambda x: int(ceil(((x) - 1) / 2)), self.kernel_size))
        else:
            dimPadding = self.dimPadding
        image_new = th.zeros(size=(dim[0], dim[1], dim[2] + dimPadding[0] * 2, dim[3] + dimPadding[1] * 2))
        image_new[:, :, dimPadding[0]:dim[2] + dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]] = image
        self.paddingFunction(image_new, dimPadding, dim)
        return image_new

    def __call__(self, dataset):
        if self.paddingFunction != 0:
            images = self.addPadding(dataset)
        else:
            images = dataset
        return self.__forward_fun__(images)

    def __forward_fun__(self, images):
        dim = list(images.shape)
        dim_new = list(map(lambda x, y, z: int(floor((x - y) / z + 1)), dim[2:4], self.kernel_size, self.stride))
        unfoldedF = nn.functional.unfold(self.filters, self.kernel_size, stride=self.stride)
        unfoldedF = th.cat(th.split(unfoldedF.unsqueeze(1), self.kernel_size[0]*self.kernel_size[1], 2), 1).unsqueeze(0)
        unfoldedIm = nn.functional.unfold(images, self.kernel_size, stride=self.stride)
        unfoldedIm = th.cat(th.split(unfoldedIm.unsqueeze(1), self.kernel_size[0]*self.kernel_size[1], 2), 1).unsqueeze(1)
        conv_output = unfoldedIm * unfoldedF
        conv_output = th.sum(th.sum(conv_output, dim=2), dim=2)  # SOMMA SUI CANALI E TRA GLI ELEMENTI DELLA FINESTRA
        conv_output = nn.functional.fold(conv_output, (dim_new[0], dim_new[1]), (1, 1))
        return conv_output

    def __str__(self):
        return f"{Fore.GREEN}ConvolutionalFilter(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.namePadding}, dimPadding={self.dimPadding}, numChannels={self.numChannels}, numFilters={self.numFilters}{Style.RESET_ALL})"


class Pooling(Model):
    def __init__(self, kernel_size, stride=(2, 2), poolingFunction='max', padding='same', dimPadding='auto'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.nameFunction = poolingFunction
        if poolingFunction == 'max':
            self.function = self.maxFunction
        elif poolingFunction == 'average':
            self.function = self.averageFunction
        elif poolingFunction == 'random':
            self.function = self.randomFunction
        elif poolingFunction == 'l2-norm':
            self.function = self.l2Function
        else:
            print(
                f"{Fore.RED}ERROR: The string \'{poolingFunction}\' does not match any pooling function implemented.\nThe options are:\n > \'max\'\n > \'average\'\n > \'random\'\n > \'l2-norm\'")
            sys.exit(-1)
        self.namePadding = padding
        if dimPadding != 'auto' and type(dimPadding) != tuple:
            print(
                f"{Fore.RED}ConvolutionalFilter: Invalid padding dimension: it must be a tuple or use the string \'auto\'{Style.RESET_ALL}")
            exit(-1)
        self.dimPadding = dimPadding
        if padding == 'zero':
            self.paddingFunction = self.zeroPadding
        elif padding == 'same':
            self.paddingFunction = self.samePadding
        elif padding == 'casual':
            self.paddingFunction = self.casualPadding
        elif padding == 'valid':
            self.paddingFunction = 0
        else:
            print(
                f"{Fore.MAGENTA}Warning: The string \'{padding}\' does not match any padding implemented, so no padding will be added.\nThe options are:\n > \'valid\'\n > \'zero\'\n > \'same\'\n > \'casual\'{Style.RESET_ALL}")
            self.paddingFunction = 0

    def zeroPadding(self, image, dimPadding, dim):
        return image

    def casualPadding(self, image, dimPadding, dim):
        rowUp = th.rand(dim[0], dim[1], dimPadding[0],  dim[3])
        rowDown = th.rand(dim[0], dim[1], dimPadding[0], dim[3])

        image[:, :, 0:dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]] = rowUp
        image[:, :, image.shape[2] - dimPadding[0]:image.shape[2], dimPadding[1]:dim[3] + dimPadding[1]] = rowDown

        columnLeft = th.rand(dim[0], dim[1], image.shape[2], dimPadding[1])
        columnRight = th.rand(dim[0], dim[1], image.shape[2], dimPadding[1])
        image[:, :, 0:image.shape[2], 0:dimPadding[1]] = columnLeft
        image[:, :, 0:image.shape[2], image.shape[3] - dimPadding[1]:image.shape[3]] = columnRight
        return image

    def samePadding(self, image, dimPadding, dim):

        rowUp = image[:, :, dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]]
        rowDown = image[:, :, dim[2], dimPadding[1]:dim[3] + dimPadding[1]]
        for i in range(0, dimPadding[0]):
            image[:, :, i, dimPadding[1]:dim[3] + dimPadding[1]] = rowUp
            image[:, :, image.shape[2] - (i + 1), dimPadding[1]:dim[3] + dimPadding[1]] = rowDown

        columnLeft = image[:, :, 0:image.shape[2], dimPadding[1]]
        columnRight = image[:, :, 0:image.shape[2], dim[3]]
        for i in range(0, dimPadding[1]):
            image[:, :, 0:image.shape[2], i] = columnLeft
            image[:, :, 0:image.shape[2], image.shape[3] - (i + 1)] = columnRight

        return image

    def addPadding(self, image):
        dim = list(image.shape)
        if self.dimPadding == 'auto':
            dimPadding = list(map(lambda x: int(ceil(((x) - 1) / 2)), self.kernel_size))
        else:
            dimPadding = self.dimPadding
        image_new = th.zeros(size=(dim[0], dim[1], dim[2] + dimPadding[0] * 2, dim[3] + dimPadding[1] * 2))
        image_new[:, :, dimPadding[0]:dim[2] + dimPadding[0], dimPadding[1]:dim[3] + dimPadding[1]] = image
        self.paddingFunction(image_new, dimPadding, dim)
        return image_new

    def parameters(self):
        return []

    def maxFunction(self, input):
        return th.max(input, dim=2)[0]

    def averageFunction(self, input):
        return th.mean(input, dim=2)

    def l2Function(self, input):
        return th.linalg.norm(input, dim=2)

    def randomFunction(self, input):
        in_dim = input.shape
        index = th.randint(in_dim[2], size=(in_dim[0], in_dim[1], 1, in_dim[3]))
        return th.take_along_dim(input, index, dim=2).squeeze(2)

    def __call__(self, dataset):
        if self.paddingFunction != 0:
            images = self.addPadding(dataset)
        else:
            images = dataset
        return self.__forward_fun__(images)

    def __forward_fun__(self, dataset):
        dim = list(dataset.shape)
        dim_new = list(map(lambda x, y, z: int((x - y) / z + 1), dim[2:4], self.kernel_size, self.stride))
        unfolded = nn.functional.unfold(dataset, self.kernel_size, stride=self.stride)
        unfolded = th.cat(th.split(unfolded.unsqueeze(1), self.kernel_size[0]*self.kernel_size[1], 2), 1)
        output = self.function(unfolded)
        pooling_output = nn.functional.fold(output, (dim_new[0], dim_new[1]), (1, 1))
        return pooling_output

    def __str__(self):
        return f"{Fore.GREEN}Pooling(kernel_size={self.kernel_size}, stride={self.stride}, poolingFunction={self.nameFunction}, padding={self.namePadding}, dimPadding={self.dimPadding}){Style.RESET_ALL}"


class Linear(Model):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.checkBias = bias
        self.weight = th.nn.Parameter(th.Tensor(out_features, in_features))
        if self.checkBias == True:
            self.bias = th.nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        th.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.checkBias == True:
            fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            th.nn.init.uniform_(self.bias, -bound, bound)

    def __call__(self, input):
        return self.__forward_fun__(input)

    def __forward_fun__(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f"{Fore.RED}\nERROR: Input size ({y}) does not match Linear() model features ({self.in_features}).\nModify the object of class Linear() or modify the input.")
            sys.exit(-1)
        output = input.matmul(self.weight.t())
        if self.checkBias == True:
            output = output + self.bias
        return output

    def parameters(self):
        if self.checkBias == True:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __str__(self):
        return f"{Fore.GREEN}Linear(input_feature={self.in_features}, output_feature={self.out_features}, bias={self.checkBias}){Style.RESET_ALL}"
