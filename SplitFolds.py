import os
import csv

def Main(numOfFold = 5):
    folder = r"<path to the folder containing the labels in csv files>"

    data = []
    with open(os.path.join(folder, "All.csv"), mode = "r") as csvFile:
        csvReader = csv.reader(csvFile)
        next(csvReader)
        for row in csvReader:
            data.append(row)

    foldDataList = []
    for fold in range(numOfFold):
        foldData = []
        for i, row in enumerate(data):
            if i % numOfFold == fold:
                foldData.append(row)
        foldDataList.append(foldData)

    for fold in range(numOfFold):
        trainSet = []
        testSet = foldDataList[fold]

        for i in range(numOfFold):
            if i != fold:
                trainSet += foldDataList[i]
        
        with open(os.path.join(folder, f"Fold{fold}Train.csv"), mode = "w", newline = "") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["ID", "Label"])
            csvWriter.writerows(trainSet)

        with open(os.path.join(folder, f"Fold{fold}Test.csv"), mode = "w", newline = "") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["ID", "Label"])
            csvWriter.writerows(testSet)

if __name__ == "__main__":
    Main()