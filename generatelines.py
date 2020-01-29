import numpy as np
import sys
import readdata
from scipy.linalg import null_space
from sympy import Matrix
from decimal import Decimal

def generatelines(pixels,K):
    alllines=[]
    # if K.shape[0]!=3 or K.shape[1]!=3:
    #    sys.exit('camera matrice is not valid it should be 3x3')
    for x, y in pixels:
        x=np.float64(x)
        y=np.float64(y)
        z = 1
        npoint = np.asarray((x, y, z))
        line=np.matmul(np.linalg.inv(K),npoint)
        line=line/np.linalg.norm(line)
        # print("**********")
        # print(np.linalg.norm(line))
        # print("-***************------")
        alllines.append(line)
    return alllines


def createlinematpython(filename,size):
    lines = readdata.read3d2('matrixlines.txt', size, 3)
    # lines = np.asarray(lines)
    for k1 in range(len(lines)):
        A = np.asarray(lines[k1,:]).T
        A=A.reshape((1,3))
        x1 = null(A)
        b = [x1, lines[k1,:]]
        c = np.asarray(x1).T
        name = 'allmatrixes/qr' + str(k1+1) + '.txt'
        np.savetxt(name,c,delimiter=',')
        # dlmwrite(name, c)
    return x1


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    # return rank,v[:].copy()
    return v[:].copy()


# def generatelines(pixels,K):
#     alllines=[]
#     # if K.shape[0]!=3 or K.shape[1]!=3:
#     #    sys.exit('camera matrix is not valid it should be 3x3')
#     for x, y in pixels:
#         x=np.float64(x)
#         y=np.float64(y)
#         z = 1
#         npoint = np.asarray((x, y, z))
#         line=np.matmul(np.linalg.inv(K),npoint)
#         line=line/np.linalg.norm(line)
#         # print("**********")
#         # print(np.linalg.norm(line))
#         # print("-***************------")
#         alllines.append(line)
#     return alllines
# # def generatelines(pixels,K):
# #     alllines=[]
# #     if K.shape[0]!=3 or K.shape[1]!=3:
# #        sys.exit('camera matrice is not valid it should be 3x3')
# #     for x, y in pixels:
# #         x=np.float64(x)
# #         y=np.float64(y)
# #         z = np.float64(K[0][0])
# #         projectedpoint = np.asarray((x, y, z))
# #         normalizenwpoint = projectedpoint / np.linalg.norm(projectedpoint)
# #         normalizenwpoint = normalizenwpoint / np.linalg.norm(normalizenwpoint[2])
# #         alllines.append(normalizenwpoint)
# #     # alllines = alllines / np.linalg.norm(alllines, axis=0)
# #     # alllines = alllines / alllines[0][2]
# #     for line in alllines:
# #         line=line/np.linalg.norm(line)
# #         # print("**********")
# #         # print (np.linalg.norm(line))
# #         # print("-***************------")
# #     pnplines = []
# #     for line in alllines:
# #         pnplines.append(line[0:2])
# #     # alllines=n,d vectors where z=1 , pnplines without z so you can use it in pnp instead of pixels ^^
# #     return alllines,pnplines
#
# def createlinemat(filename,eng):
#     print('create lines using matlab')
#     # eng.generateliness(filename, nargout=0)
#     eng.generateliness
#
#
# def createlinematpython(filename,size):
#     lines = readdata.read3d2('matrixlines.txt', size, 3)
#     # lines = np.asarray(lines)
#     for k1 in range(len(lines)):
#         A = np.asarray(lines[k1,:]).T
#         A=A.reshape((1,3))
#         x1 = null(A)
#         b = [x1, lines[k1,:]]
#         c = np.asarray(x1).T
#        # name = 'C:\Users\firas\PycharmProjects\backprojection\allmatrixes\qr'+ str(k1), '.txt')
#         name = '/home/fares/PycharmProjects/Point-Tracker/allmatrixes/qr' + str(k1+1) + '.txt'
#         np.savetxt(name,c,delimiter=',')
#         # dlmwrite(name, c)
#     return x1
#
#
# def null(a, rtol=1e-5):
#     u, s, v = np.linalg.svd(a)
#     rank = (s > rtol*s[0]).sum()
#     # return rank,v[:].copy()
#     return v[:].copy()

# createlinematpython(0,100)