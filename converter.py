import numpy as np
import csv
import cv2 as cv

POINTFILE = "pointData.csv"
DESCFILE = "descriptorsData.xml"


def load_points():
    points = list()
    with open(POINTFILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            points.append((row[0], row[1], row[2]))
    return np.array(points, dtype=np.float32)


def load_descriptors():
    descriptors = list()
    i = 0
    fs = cv.FileStorage(DESCFILE, cv.FILE_STORAGE_READ)
    while True:
        fn = fs.getNode("desc" + str(i))
        if fn.mat() is not None:
            descriptors.append(fn.mat())
            i += 1
        else:
            return np.array(descriptors, dtype=np.uint8)


def main():
    points = load_points()

    np.save(r"Data/points.csv", points)
    descriptors = load_descriptors()

    np.save(r"Data/descriptors.csv", descriptors)
    print(len(points))
    print(len(descriptors))

    print(points)
    print(descriptors)


if __name__ == "__main__":
    main()
