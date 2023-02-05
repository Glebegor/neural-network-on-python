# import numpy as np
# import random
# import gzip
# import matplotlib.pyplot as plt

# # Adding MINIST DataSet
# datasetImg = gzip.open("train-images-idx3-ubyte.gz", "r")
# datasetLable = gzip.open("train-labels-idx1-ubyte.gz", "r")

# image_size = 28
# num_images = 6001

# datasetImg.read(16)
# datasetLable.read(8)

# bufImg = datasetImg.read(image_size * image_size * num_images)
# bufLable = datasetLable.read(num_images)

# dataImg = np.frombuffer(bufImg, dtype=np.uint8).astype(np.float32)
# dataLable = np.frombuffer(bufLable, dtype=np.uint8).astype(np.float32)
# # End Adding MINIST DataSet

# input_number = 784
# neurons_number = 128
# output_number = 10

# # Active Func
# def sigmoid(x, derivative=False):
#       if derivative:
#           return (np.exp(-x))/((np.exp(-x)+1)**2)
#       return 1/(1 + np.exp(-x))
# def softmax(x, derivative=False):
#     # Numerically stable with large exponentials
#     exps = np.exp(x - x.max())
#     if derivative:
#         return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
#     return exps / np.sum(exps, axis=0)
# # End Active Func

# def throw_in(Firs_W, Sec_W, Thr_W, arrrrs):
#     sum1 = np.dot(Firs_W, arrrrs)
#     res1 = sigmoid(sum1)
    
#     sum2 = np.dot(Sec_W, res1)
#     res2 = sigmoid(sum2)

#     sum3 = np.dot(Thr_W, res2)
#     out = softmax(sum3)

#     return (sum1, res1, sum2, res2, sum3, out)

# def training_NeuNet():
#     global W1
#     global W2
#     global W3
#     data = dataImg.reshape(num_images, image_size, image_size, 1)
#     lmd = 0.001
#     learning_rate = 0.15
#     N = 10000

#     for g in range(N):
#         RandidImg = np.random.randint(0,6001)
#         ar = (np.asfarray(data[RandidImg])/255*0.99)+0.01

#         echopMain = np.array([])
#         for i in range(0,len(ar)):
#             for j in range(0, len(ar[i])):
#                 echopMain = np.append(echopMain, ar[i][j][0])

#         solution = int(dataLable[RandidImg])
#         solve = []
#         for i in range(0, solution+1):
#             if i == solution:
#                 solve.append(1)
#                 for j in range(len(solve), 10):
#                     solve.append(0)
#             else:
#                 solve.append(0)

#         sum1, res1, sum2, res2, sum3, out = throw_in(Firs_W=W1, Sec_W=W2, Thr_W=W3, arrrrs=echopMain)
        
#         Errors = {}
#         err = 2*(out - solve) / out.shape[0]*softmax(sum3, derivative=True)
#         Errors["W3"] = np.outer(err, res2)
#         print(Errors["W3"])
#         W3 -= lmd * Errors["W3"]


#         err = np.dot(W3.T, err)*sigmoid(sum2, derivative=True)
#         Errors["W2"] = np.outer(err, res1)
#         W2 -= lmd * Errors["W2"]


#         err = np.dot(W2.T, err)*sigmoid(sum1, derivative=True)
#         Errors["W1"] = np.outer(err, echopMain)
#         W1 -= lmd * Errors["W1"]

#         print("Number: ", solution)
#         print("Must be: ", solve)
#         # print("Result1: ", res1)
#         print("Out: ", out)

#         # image = np.asarray(data[RandidImg]).squeeze()
#         # plt.imshow(image)
#         # plt.show()

# W1 = np.random.randn(128, 784) * np.sqrt(1. / 128)
# W2 = np.random.randn(64, 128) * np.sqrt(1. / 64)
# W3 = np.random.randn(10, 64) * np.sqrt(1. / 10)
# print(W3)
# training_NeuNet()
