import time
import torch

from Utility import AverageMeter, TopKAccuracy, AUC

def Train(trainLoader, net, optimizer, lossFunction, epoch, rank, mode = "multiple"):
    """
    Train one epoch with given data loader, network, optimizer and loss function

    Parameters
    ----------
    trainLoader : torch.utils.data.DataLoader

    net : torch.nn.Module
        Should be on cuda device already
    
    optimizer : torch.optim.Optimizer

    lossFunction : torch.nn.Module

    epoch : int
        The current epoch, used for logging

    rank : int
        The rank of current process, used for logging

    mode : str, default = "multiple"
        multiple or single
    """
    batchTime = AverageMeter("BatchTime")
    dataTime = AverageMeter("DataTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")

    net.train()

    startTime = time.perf_counter_ns()
    for batchData, batchLabel, _ in trainLoader:
        dataTime.Update((time.perf_counter_ns() - startTime) / 1e6)

        batchData = batchData.cuda()
        batchLabel = batchLabel.cuda()

        batchPredict = net(batchData)
        loss = lossFunction(batchPredict, batchLabel)

        losses.Update(loss.item(), batchData.size(0))
        top1accuracy = TopKAccuracy(batchPredict, batchLabel, topk = (1,))
        top1Accuracies.Update(top1accuracy[0], batchData.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

    if mode == "multiple":
        batchTime.AllReduce()
        dataTime.AllReduce()
        losses.AllReduce()
        top1Accuracies.AllReduce()

    if mode == "single":
        top1Accuracies.Average = top1Accuracies.Average.item()

    if rank == 0 or mode == "single":
        toPrint = f"\nEpoch [{epoch}]\n"
        toPrint += f"Train:      [loss, accuray] -> [{losses.Average:.4e}, {top1Accuracies.Average:.2f}%], "
        toPrint += f"[bacth time, data time] -> [{batchTime.Average:.6f}ms, {dataTime.Average:.6f}ms]"
        print(toPrint)

    return losses.Average, top1Accuracies.Average

def EvaluateSingle(testLoader, net, lossFunction, name):
    """
    Evaluate the network with given data loader and loss function

    Parameters
    ----------
    testLoader : torch.utils.data.DataLoader

    net : torch.nn.Module
        Should be on cuda device already

    lossFunction : torch.nn.Module

    name : str
        Should be "Test" or "Validation"
    """
    batchTime = AverageMeter("BatchTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")

    def EvaluateLoop(loader):
        batchPredictions = []
        batchLabels = []
        with torch.no_grad():
            startTime = time.perf_counter_ns()
            for batchData, batchLabel, _ in loader: 
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()

                # print(f"Test process got {len(batchData)} samples!")

                batchPrediction = net(batchData)
                loss = lossFunction(batchPrediction, batchLabel)

                losses.Update(loss.item(), batchData.size(0))
                top1accuracy = TopKAccuracy(batchPrediction, batchLabel, topk = (1,))
                top1Accuracies.Update(top1accuracy[0], batchData.size(0))

                batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

                # Here, the loader is assumed to be not shuffled
                # If shuffled or any disorder appears, you should use the index (the third return value of the loader)
                batchPredictions.append(batchPrediction)
                batchLabels.append(batchLabel)

        return batchPredictions, batchLabels

    net.eval()
    batchPredictions, batchLabels = EvaluateLoop(testLoader)

    """
    In single GPU mode, as long as the drop_last of DataLoader is set to False
    The above loop should handle all samples in the dataset, for example:
    15 data samples in the dataloader and batch size is 6, then the loop will be run 3 times with:
    Batch 0: 6 samples
    Batch 1: 6 samples
    Batch 2: 3 samples
    """
    predictions = torch.cat(batchPredictions, dim = 0)
    labels = torch.cat(batchLabels, dim = 0)

    if hasattr(top1Accuracies.Average, "item"):
        top1Accuracies.Average = top1Accuracies.Average.item()
    auc = AUC(predictions, labels)

    toPrint = ""
    toPrint += f"{name}:{' ' if len(name) > 4 else '       '}[loss, accuray, auc] -> [{losses.Average:.4e}, {top1Accuracies.Average:.2f}%, {auc:.3f}], "
    toPrint += f"bacth time -> {batchTime.Average:.6f}ms"
    print(toPrint)
        
    return losses.Average, top1Accuracies.Average, predictions.cpu(), labels.cpu()

def EvaluateMultiple(testLoader, net, lossFunction, name, rank, worldSize):
    """
    Evaluate the network with given data loader and loss function

    Parameters
    ----------
    testLoader : torch.utils.data.DataLoader

    net : torch.nn.Module
        Should be on cuda device already

    lossFunction : torch.nn.Module

    name : str
        Should be "Test" or "Validation"

    rank : int
        The rank of current process, used for logging

    worldSize : int
        The number of processes, used for logging
    """
    batchTime = AverageMeter("BatchTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")

    def EvaluateLoop(loader):
        batchIndexes = []
        batchPredictions = []
        with torch.no_grad():
            startTime = time.perf_counter_ns()
            for batchData, batchLabel, batchIndex in loader:
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()
                batchIndex = batchIndex.cuda()

                # print(f"GPU {rank} got {len(batchData)} samples")

                batchPrediction = net(batchData)
                loss = lossFunction(batchPrediction, batchLabel)

                losses.Update(loss.item(), batchData.size(0))
                top1accuracy = TopKAccuracy(batchPrediction, batchLabel, topk = (1,))
                top1Accuracies.Update(top1accuracy[0], batchData.size(0))

                batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

                batchIndexes.append(batchIndex)
                batchPredictions.append(batchPrediction)

        return batchIndexes, batchPredictions

    net.eval()
    batchIndexes, batchPredictions = EvaluateLoop(testLoader)

    batchTime.AllReduce()
    losses.AllReduce()
    top1Accuracies.AllReduce()

    indexes = []
    predictions = []
    for batchIndex, batchPrediction in zip(batchIndexes, batchPredictions):
        gatheredBatchIndex = [torch.zeros_like(batchIndex) for _ in range(worldSize)]
        torch.distributed.all_gather(gatheredBatchIndex, batchIndex, async_op = False)
        indexes.append(torch.cat(gatheredBatchIndex, dim = 0))

        gatheredBatchPrediction = [torch.zeros_like(batchPrediction) for _ in range(worldSize)]
        torch.distributed.all_gather(gatheredBatchPrediction, batchPrediction, async_op = False)
        predictions.append(torch.cat(gatheredBatchPrediction, dim = 0))

    indexes = torch.cat(indexes, dim = 0)
    predictions = torch.cat(predictions, dim = 0)

    if len(testLoader.sampler) * worldSize < len(testLoader.dataset):
        auxiliaryTestDataset = torch.utils.data.Subset(
            testLoader.dataset,
            range(len(testLoader.sampler) * worldSize, len(testLoader.dataset))
        )
        auxiliaryTestLoader = torch.utils.data.DataLoader(
            auxiliaryTestDataset, batch_size = testLoader.batch_size, shuffle = False,
            num_workers = testLoader.num_workers, persistent_workers = testLoader.persistent_workers, pin_memory = True
        )
        auxiliaryBatchIndexes, auxiliaryBatchPredictions = EvaluateLoop(auxiliaryTestLoader)

        indexes = torch.cat([indexes] + auxiliaryBatchIndexes, dim = 0)
        predictions = torch.cat([predictions] + auxiliaryBatchPredictions, dim = 0)

    predictions = predictions[indexes.argsort()]

    if hasattr(top1Accuracies.Average, "item"):
        top1Accuracies.Average = top1Accuracies.Average.item()

    if rank == 0:
        toPrint = ""
        toPrint += f"{name}:{' ' if len(name) > 4 else '       '}[loss, accuray] -> [{losses.Average:.4e}, {top1Accuracies.Average:.2f}%], "
        toPrint += f"bacth time -> {batchTime.Average:.6f}ms"
        print(toPrint)

    return losses.Average, top1Accuracies.Average, predictions

def Evaluate(testLoader, net, lossFunction, name, rank, worldSize, mode):
    """
    Evaluate the network with given data loader and loss function

    The evaluation process in DDP can be very tricky
    therefore the evaluation functions of these two mode are separated for easier debugging

    Parameters
    ----------
    The other parameters are expained in the function EvaluateSingle and EvaluateMultiple

    mode : str, default = "multiple"
        multiple or single
    """
    if mode == "single":
        return EvaluateSingle(testLoader, net, lossFunction, name)
    if mode == "multiple":
        return EvaluateMultiple(testLoader, net, lossFunction, name, rank, worldSize)
    raise Exception(f"Bad parameter \"mode\": {mode}")
