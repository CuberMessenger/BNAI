import os
import sys

import time
import shutil
import argparse
import torch
import torch.nn as nn
import Network

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders
from Utility import Result, StandardOutputDuplicator, AUC
from Train import GetNet

OriginalSystemStandardOutput = sys.stdout

def TestWorker(configuration, logFile, fold):
    """
    Worker function for single GPU testing
    """
    sys.stdout = StandardOutputDuplicator(OriginalSystemStandardOutput, logFile)

    net = GetNet(configuration).cuda()
    net.load_state_dict(torch.load(os.path.join(configuration["WeightFolder"], f"Weights-Fold{fold}.pkl")))

    lossFunction = nn.CrossEntropyLoss().cuda()

    trainLoader, _, testLoader, externalTestLoader = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = False,
        fold = fold
    )

    trainLoss, trainAccuracy, trainPredictions, trainLabels = Evaluate(
        trainLoader, net, lossFunction,
        "Train", 0, 0, mode = "single"
    )

    auc, threshold = AUC(trainPredictions, trainLabels, needThreshold = True)
    print(f"Train AUC: {auc}, Threshold: {threshold}")
    print()

    print("Test set:")
    testLoss, testAccuracy, testPredictions, testLabels = Evaluate(
        testLoader, net, lossFunction,
        "Test", 0, 0, mode = "single"
    )
    for id, label, prediction in zip(testLoader.dataset.Dataset.IDs, testLabels, testPredictions.softmax(dim = 1)[: , 1].cpu()):
        print(f"{id},{label},{prediction.item()}")

    externalTestLoss, externalTestAccuracy, externalTestPredictions, externalTestLabels = Evaluate(
        externalTestLoader, net, lossFunction,
        "ExternalTest", 0, 0, mode = "single"
    )

    results = {
        "TestLoss": testLoss,
        "TestAccuracy": testAccuracy,
        "TestPredictions": testPredictions,
        "ExternalTestLoss": externalTestLoss,
        "ExternalTestAccuracy": externalTestAccuracy,
        "ExternalTestPredictions": externalTestPredictions
    }

    torch.save(results, os.path.join(configuration["SaveFolder"], f"TestResults-Fold{fold}.pt"))

def PostAnalysis(configuration):
    # Train set overall AUC by cross-validation test set 
    labels = torch.tensor([])
    predictions = torch.tensor([])
    for fold in range(5):
        _, _, testLoader, _ = GetDataLoaders(
            configuration["DatasetName"],
            configuration["BatchSize"],
            configuration["NumOfWorker"],
            configuration["DataFolder"],
            distributed = False,
            fold = fold
        )
        results = torch.load(os.path.join(configuration["SaveFolder"], f"TestResults-Fold{fold}.pt"))
        labels = torch.cat([labels, torch.tensor(testLoader.dataset.Dataset.Labels)])
        predictions = torch.cat([predictions, results["TestPredictions"]])

    auc = AUC(predictions, labels)
    predictions = predictions.argmax(dim = 1)
    accuracy = (predictions == labels).float().mean().item()
    print(f"Overall (Accuracy, AUC) on 5 test sets: ({accuracy}, {auc})")

    # Ensemble cross-validation's prediction on external test set
    _, _, _, externalTestLoader = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = False,
        fold = fold
    )
    externalLabels = torch.tensor(externalTestLoader.dataset.Dataset.Labels)
    externalPredictions = torch.tensor([])
    for fold in range(5):
        results = torch.load(os.path.join(configuration["SaveFolder"], f"TestResults-Fold{fold}.pt"))
        externalPredictions = torch.cat([externalPredictions, results["ExternalTestPredictions"][None, ...]])
    externalPredictions = externalPredictions.mean(dim = 0).softmax(dim = 1)

    auc = AUC(externalPredictions, externalLabels)
    accuracy = (externalPredictions.argmax(dim = 1) == externalLabels).float().mean().item()
    print(f"Ensemble (Accuracy, AUC) on external test set: ({accuracy}, {auc})")

    for id, label, prediction in zip(externalTestLoader.dataset.Dataset.IDs, externalLabels, externalPredictions[:, 1]):
        print(f"{id},{label},{prediction.item()}")

def Main(configuration):
    if not os.path.exists(configuration["ResultFolder"]):
        os.mkdir(configuration["ResultFolder"])

    folderName = os.path.basename(configuration["WeightFolder"])
    if folderName == "":
        # If the parsed path looks like "a/b/c/", the basename will be ""
        folderName = os.path.basename(os.path.dirname(configuration["WeightFolder"]))

    saveFolder = os.path.join(
        configuration["ResultFolder"],
        f"{folderName}-Test-{time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())}"
    )
    os.mkdir(saveFolder)
    configuration["SaveFolder"] = saveFolder

    for fold in range(5):
        try:
            with open(os.path.join(configuration["SaveFolder"], f"Log-Fold{fold}.txt"), mode = "w") as logFile:
                TestWorker(configuration, logFile, fold)
        except Exception as e:
            # If any exception occurs, delete the save folder
            shutil.rmtree(configuration["SaveFolder"])
            raise e
        else:
            sys.stdout = OriginalSystemStandardOutput

    with open(os.path.join(configuration["SaveFolder"], f"PostAnalysis.txt"), mode = "w") as logFile:
        sys.stdout = StandardOutputDuplicator(OriginalSystemStandardOutput, logFile)
        PostAnalysis(configuration)
    # Log the source code
    shutil.copy(
        __file__,
        os.path.join(configuration["SaveFolder"], "TestScript.py")
    )

if __name__ == "__main__":
    configuration = {
        "NumOfWorker": 4,
        "BatchSize": 16,
        "NetName": "ResNet18Pretrained",
        "DatasetName": "BreastUltrasound",
        "DataFolder": r"<path to the folder containing data>",
        "ResultFolder": r"<path to the folder containing results>"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-folder", type = str)
    args = parser.parse_args()

    configuration["WeightFolder"] = args.weight_folder
    if configuration["WeightFolder"] is None:
        configuration["WeightFolder"] = r"<path to a result folder such as ...\ResNet18Pretrained-BreastUltrasound-%Y-%m-%d-%H.%M.%S>"

    Main(configuration)
