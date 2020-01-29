import numpy as np
import scipy.io
import os

def write_data_python(name, array):
    """
    :param name:  the location for array to be saved in - put .npy in the end just to be sure
    :param array: the nd array
    """
    np.save(name, array)


def write_data_matlab(name, array):
    """
    :param name: the location for array to be saved in - should be ended with  .mat name
    :param array:
    """
    scipy.io.savemat(name, dict(array=array))


def read_data_python(name):
    """
    :param name: name of the file to read from. should be ended with .npy
    :return: data as nd array
    """
    return np.load(name)

def read_ortho(name):
    ortho=np.zeros((3,3))
    file = open("/home/fares/PycharmProjects/Point-Tracker/allmatrixes/qr"+str(name)+".txt", "r")

    i=0
    for line in file:
        a = line.split(',')
        ortho[i,:]=[float(a[0]),float(a[1]),float(a[2])]
        i+=1
    return ortho


# *******************      Could be Used in CPP.run   ********************
def readpixels2(name,datasize,d):
    file = open(name, "r")
    pixels=np.zeros((d-1,datasize))
    i=0
    for line in file:
        pixar = []
        a=line.split(' ')
        for p in a:
            # if len(p)<3:
            #     break
            b=float(p)
            pixar.append(b)
        pixarnp=np.asarray(pixar)
        pixels[:,i]=pixarnp
        i+=1
    pixels=np.transpose(pixels)
    return pixels


# *******************      Could be Used after CPP.run    ********************
def read3d2(name,datasize,d):
    file = open(name, "r")
    threed=np.zeros((d,datasize))
    i=0
    for line in file:
        pointar = []
        a=line.split(' ')
        for p in a:
            # if len(p)<3:
            #     break
            b=float(p)
            pointar.append(b)
        pointarnp=np.asarray(pointar)
        threed[:,i]=pointarnp
        i+=1
    threed=np.transpose(threed)
    return threed


def creatdirectories(OutputFileName,datasize):

    # os.makedirs(OutputFileName)
    os.makedirs(OutputFileName + '/text')
    os.makedirs(OutputFileName + '/mat')
    os.makedirs(OutputFileName + '/npy')
    # os.makedirs(OutputFileName + '/text/epsilon')
    # os.makedirs(OutputFileName + '/mat/epsilon')
    # os.makedirs(OutputFileName + '/npy/epsilon')
    # os.makedirs('{}\{}\{}.txt'.format(a, b, c))


# if I want to read in matlab i just write load <name> so I don't need func for this.\
def savedata(OutputFileName,datasize,normalized_lines,points):
        write_data_python(OutputFileName + '/npy/lines.npy', normalized_lines)
        write_data_matlab(OutputFileName + '/mat/lines.mat', normalized_lines)
        write_data_python(OutputFileName + '/npy/points.npy', points)
        write_data_matlab(OutputFileName + '/mat/points.mat', points)

        linefile = open(OutputFileName + "/text/lines.txt", "w")
        pointfile = open(OutputFileName + "/text/points.txt", "w")
        for i in range(len(normalized_lines)):
                linefile.write(str(float(normalized_lines[i][0]))+' '+str(float(normalized_lines[i][1]))+' '+str(float(normalized_lines[i][2]))+'\n')
                pointfile.write(str(float(points[i][0]))+' '+str(float(points[i][1]))+' '+str(float(points[i][2]))+'\n')

        linefile.close()
        pointfile.close()


