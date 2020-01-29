import numpy as np
import readdata
import generatelines
import PnP_coreset
import os
# import matlab.engine
import sys
import shutil
import scipy.io

def run(size,points3D,points2D,Lines,rotationMat=None,T=None,linefilename=None,OutputFileName=None):
    runepsilon = True
    linefilename = "matrixlines.txt"
    linefile = open(linefilename, "w")
    # OutputFileName = 'InputForMatlab'
    # DataSize = len(pattern)

    # eng = matlab.engine.start_matlab()

    np.asarray(points2D, dtype=np.float64)
    np.asarray(points3D, dtype=np.float64)
    alllines = np.asarray(Lines)

    for i in range(size):
        linefile.write(str(float(alllines[i][0])) + ' ' + str(float(alllines[i][1])) + ' ' + str(float(alllines[i][2])) + '\n')
    linefile.close()

    # this function calculates and saves each line matrix in a file called "allmatrixes"
    generatelines.createlinematpython(linefilename, size)
    # if os.path.exists(OutputFileName):
    #     shutil.rmtree(OutputFileName,ignore_errors=True)
    # os.makedirs(OutputFileName)
    # for i in range(size):
    #     readdata.creatdirectories(OutputFileName, i)
    #     readdata.savedata(OutputFileName, i, alllines, points3D)

    if runepsilon:
        # create epsilon sample coreset
        indexArray, weightArray = PnP_coreset.createcoresets(size, 3, np.asarray(points3D))

        # Return indexes and weights
        return indexArray, weightArray

    return None, None


if __name__ == "__main__":
    run()