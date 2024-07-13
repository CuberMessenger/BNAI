import os
import sys

import time
import shutil
import torch
import torch.nn as nn
import Network

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders
from Utility import Result, StandardOutputDuplicator

OriginalSystemStandardOutput = sys.stdout

def GetNet(configuration):
    """
    Return the network according to the dataset
    """
    if configuration["DatasetName"] == "BreastUltrasound":
        inChannels = 1
        numOfClass = 2
        parameters = inChannels, numOfClass
    else:
        raise ValueError(f"Unknown dataset name {configuration['DatasetName']}")

    net = getattr(Network, configuration["NetName"])(*parameters)
    return net

def TrainWorker(configuration, logFile, fold):
    """
    Worker function for single GPU training
    """
    sys.stdout = StandardOutputDuplicator(OriginalSystemStandardOutput, logFile)

    net = GetNet(configuration).cuda()

    lossFunction = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr = configuration["LearnRate"],
    )
    scheduler = None

    # A injector can be injected into the beginning of each epoch with parameters (net, epoch)
    injector = None

    trainLoader, validationLoader, testLoader, _ = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = False,
        fold = fold
    )

    bestEpoch = 1
    bestIndicator = 0x7FFF
    result = Result()

    for epoch in range(1, configuration["NumOfEpoch"] + 1):
        # Do possible injector for controlling net during training
        if injector is not None:
            injector(net, epoch)
 
        # Train one epoch
        trainLoss, trainAccuracy = Train(
            trainLoader, net, optimizer,
            lossFunction, epoch, 0, mode = "single"
        )

        # Do evaluation (some dataset has no validation set)
        if len(validationLoader.dataset) > 0:
            validationLoss, validationAccuracy, _, _ = Evaluate(
                validationLoader, net, lossFunction,
                "Validation", 0, 0, mode = "single"
            )
        else:
            validationLoss, validationAccuracy = None, None

        # Do evaluation on test set
        # Set testLoss and testAccuracy to None if you don't want to do it every epoch
        testLoss, testAccuracy, _, _ = Evaluate(
            testLoader, net, lossFunction,
            "Test", 0, 0, mode = "single"
        )

        # Step the learn rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging and saving the best model to the save folder on the main process
        result.Append(
            trainLoss, trainAccuracy,
            validationLoss, validationAccuracy,
            testLoss, testAccuracy
        )

        if len(validationLoader.dataset) > 0:
            indicator = validationLoss
        else:
            indicator = trainLoss

        if indicator < bestIndicator:
            bestEpoch = epoch
            torch.save(net.state_dict(), os.path.join(configuration["SaveFolder"], f"Weights-Fold{fold}.pkl"))
        bestIndicator = min(bestIndicator, indicator)

    print(f"Best epoch -> {bestEpoch}")
    return result
        
def Main(configuration):
    if not os.path.exists(configuration["ResultFolder"]):
        os.mkdir(configuration["ResultFolder"])

    saveFolder = os.path.join(
        configuration["ResultFolder"],
        f"{configuration['NetName']}-{configuration['DatasetName']}-{time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())}"
    )
    os.mkdir(saveFolder)
    configuration["SaveFolder"] = saveFolder

    for fold in range(5):
        try:
            with open(os.path.join(configuration["SaveFolder"], f"Log-Fold{fold}.txt"), mode = "w") as logFile:
                result = TrainWorker(configuration, logFile, fold)
        except Exception as e:
            # If any exception occurs, delete the save folder
            shutil.rmtree(configuration["SaveFolder"])
            raise e
        else:
            result.Save(os.path.join(configuration["SaveFolder"], f"Result-Fold{fold}.txt"))

    # Log the source code
    shutil.copy(
        __file__,
        os.path.join(configuration["SaveFolder"], "TrainScript.py")
    )

if __name__ == "__main__":
    configuration = {
        "NumOfWorker": 4,
        "LearnRate": 2e-5,
        "BatchSize": 16,
        "NumOfEpoch": 50,
        "NetName": "ResNet18Pretrained",
        "DatasetName": "BreastUltrasound",
        "DataFolder": r"<path to the folder containing data>",
        "ResultFolder": r"<path to the folder containing results>"
    }
    Main(configuration)
