import os
import cv2
import csv
import glob
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def BoundingBox(image):
    rows = np.any(image, axis = 1)
    columns = np.any(image, axis = 0)
    rowMin, rowMax = np.where(rows)[0][[0, -1]]
    columnMin, columnMax = np.where(columns)[0][[0, -1]]

    return columnMin, columnMax, rowMin, rowMax

def EqualizeHistogram(image):
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    image = clahe.apply(image)
    return image

class BreastUltrasoundDataset(Dataset):
    def __init__(self, root, name, transform = None):
        """
        - root: a folder containing a csv and a folder of images and ROIs
        """
        self.IDs = []
        self.Labels = []

        with open(os.path.join(root, "Labels", f"{name}.csv"), mode = "r") as csvFile:
            csvReader = csv.reader(csvFile)
            next(csvReader)
            for row in csvReader:
                self.IDs.append(row[0])

                label = None
                if row[1] == "malignant":
                    label = 1
                if row[1] == "benign":
                    label = 0
                if label is None:
                    raise Exception(f"Bad label of {row[0]}")

                self.Labels.append(label)
        
        self.Root = root
        self.Transform = transform

        if os.path.isfile(os.path.join(root, "Preprocessed.pt")):
            self.Images = torch.load(os.path.join(root, "Preprocessed.pt"))
        else:
            self.Images = self.Preprocess()

    def Preprocess(self):
        """
        Preprocess the images and save them to a file
        """
        ids = []
        for name in ["All", "ExternalTest"]:
            with open(os.path.join(self.Root, "Labels", f"{name}.csv"), mode = "r") as csvFile:
                csvReader = csv.reader(csvFile)
                next(csvReader)
                for row in csvReader:
                    ids.append(row[0])
        
        data = {}
        for id in ids:
            imagePath = glob.glob(os.path.join(self.Root, "Images", f"{id}.*"))[0]
            maskPath = os.path.join(self.Root, "Images", f"{id}-1.tif")

            image = np.array(Image.open(imagePath).convert("L"))
            mask = np.array(Image.open(maskPath).convert("L"))

            if mask[0, 0] == 255:
                mask = 255 - mask

            columnMin, columnMax, rowMin, rowMax = BoundingBox(mask)
            width = columnMax - columnMin
            height = rowMax - rowMin

            if width >= height:
                padding = (width - height) // 2
                rowMin = max(0, rowMin - padding)
                rowMax = min(mask.shape[0], rowMax + padding)
            else:
                padding = (height - width) // 2
                columnMin = max(0, columnMin - padding)
                columnMax = min(mask.shape[1], columnMax + padding)

            image = image[rowMin:rowMax, columnMin:columnMax]
            image = EqualizeHistogram(image)
            image = Image.fromarray(image).resize((224, 224))
            
            image.save(os.path.join(self.Root, "Preprocessed", f"{id}.png"))

            image = np.array(image)
            image = torch.tensor(image, dtype = torch.float32)
            image /= 255
            image = image[None, ...] # 1 x 224 x 224

            data[id] = image

        torch.save(data, os.path.join(self.Root, "Preprocessed.pt"))
        return data
            
    def __getitem__(self, index):
        id = self.IDs[index]
        image = self.Images[id]
        label = self.Labels[index]

        if self.Transform:
            image = self.Transform(image)

        return image, label
    
    def __len__(self):
        return len(self.IDs)

class DatasetWithIdentity(Dataset):
    """
    A dataset class that returns both the sample and its identity
    If you wrap a dataset with this class, the returned tuple will be (sample, identity, index)
    It is convenient to use this class to get the index of a sample, especially in DDP scneario
    """
    def __init__(self, dataset: Dataset):
        self.Dataset = dataset

    def __len__(self):
        return len(self.Dataset)

    def __getitem__(self, index):
        toReturn = self.Dataset.__getitem__(index)
        toReturn = toReturn + (index,)
        return toReturn

def GetDataLoaders(datasetName, batchSize, numOfWorker, saveFolder, distributed = False, fold = 0):
    """
    Return train, validation and test data loaders
    If no validation set is available, the validation loader is a loader object with no data
    
    Parameters
    ----------
    datasetName : str
        BreastUltrasound

    batchSize : int

    numOfWorker : int

    saveFolder : str
        Path to the folder to save the datasets

    distributed : bool, default False
        If True, the data loaders are created with distributed samplers
    """
    if datasetName == "BreastUltrasound":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees = 15, translate = (0.1, 0.1), scale = (0.9, 1.1), shear = 15),
            transforms.ToTensor()
        ])
        trainSet = BreastUltrasoundDataset(saveFolder, f"Fold{fold}Train", transform = transform)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [len(trainSet) - 52, 52])
        testSet = BreastUltrasoundDataset(saveFolder, f"Fold{fold}Test")
        externalTestSet = BreastUltrasoundDataset(saveFolder, f"ExternalTest")
    else:
        raise Exception(f"Unknown dataset name {datasetName}")

    trainSet = DatasetWithIdentity(trainSet)
    validationSet = DatasetWithIdentity(validationSet)
    testSet = DatasetWithIdentity(testSet)
    externalTestSet = DatasetWithIdentity(externalTestSet)

    persistentWorkers = numOfWorker > 0
    trainSampler, validationSampler, testSampler, externalTestSampler = None, None, None, None
    if distributed:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet, drop_last = True)
        validationSampler = torch.utils.data.distributed.DistributedSampler(validationSet, shuffle = False, drop_last = True)
        testSampler = torch.utils.data.distributed.DistributedSampler(testSet, shuffle = False, drop_last = True)
        externalTestSampler = torch.utils.data.distributed.DistributedSampler(externalTestSet, shuffle = False, drop_last = True)
    trainLoader = torch.utils.data.DataLoader(
        trainSet, batch_size = batchSize, shuffle = not distributed, sampler = trainSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )
    validationLoader = torch.utils.data.DataLoader(
        validationSet, batch_size = batchSize, sampler = validationSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )
    testLoader = torch.utils.data.DataLoader(
        testSet, batch_size = batchSize, sampler = testSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )
    externalTestLoader = torch.utils.data.DataLoader(
        externalTestSet, batch_size = batchSize, sampler = testSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )

    if distributed:
        return trainLoader, validationLoader, testLoader, externalTestLoader, trainSampler
    else:
        return trainLoader, validationLoader, testLoader, externalTestLoader
