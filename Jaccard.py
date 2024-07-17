
import numpy as np
import time
import torch


def Jaccard(MatrixAdjacency_Train):
    similarity_StartTime = time.clock()

    MatrixAdjacency_Train = torch.FloatTensor(MatrixAdjacency_Train).cuda()

    MatrixAdjacency_Train = MatrixAdjacency_Train.cpu()
    # print(MatrixAdjacency_Train.shape)
    MatrixAdjacency_Train = MatrixAdjacency_Train.numpy()
    deg_row = sum(MatrixAdjacency_Train)
    # print(type(deg_row.shape[0]))
    # num_shape = deg_row.shape[0]
    deg_row.shape = (deg_row.shape[0], 1)
    deg_row_T = deg_row.T
    tempdeg = deg_row + deg_row_T

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    temp = tempdeg - Matrix_similarity

    Matrix_similarity = Matrix_similarity / temp
    print(Matrix_similarity.shape)

    return Matrix_similarity

