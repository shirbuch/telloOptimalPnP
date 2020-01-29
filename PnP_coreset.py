import numpy as np
import scipy.linalg as LA
import readdata
import os
import time

# P is a set of points (np array of size nXd)
# V is an nparray of size n(d-1)Xd, which contains in its rows the orthogonal complement of each line, for example:
#   rows 0-(d-2) are the d-1 orthogonal complements of l1
#   rows (d-1)-(2d-3) are the d-1 orthogonal complements of l2
# B is a set of (d-1)-dimensional translation vectors (an nparray of size nX(d-1)) for each of the input n lines
def createcoresets(N,D,points):
    def generate_data(P, V, B):
        n = P.shape[0]
        d = P.shape[1]
        S = np.zeros((d**2+d+1,n*(d-1)))
        for i in range(n):
            p_curr = P[i,:]
            V_curr = V[i*(d-1):(i+1)*(d-1),:]
            b_curr = B[i,:]
            S_i = np.zeros((d**2+d+1,d-1))
            # Compute s_1,...,s_{d-1} (s_0,...,s_{d-2}) for every input pair
            for k in range(d-1):
                # Compute the vector s_k = (V[k,0]*p,...,V[k,d-1]*p,V[k,:],b[k])
                s_k = np.concatenate((np.outer(V_curr[k,:], p_curr).reshape(-1),V_curr[k,:].reshape(-1),b_curr[k].reshape(-1)),axis=0)
                S_i[:,k] = s_k
            S[:,i*(d-1):(i+1)*(d-1)] = S_i
        return S

    # Implement the epsilon (sensitivity) coreset from the paper.
    # If S = UDV^T is the svd of the matrix S computed using generate_data, then the sensitivity of the i'th point is the norm
    # # of the rows of U corresponding to the i'th point.
    def PnP_eps_coreset(P, V, B, C_size=None):
        n = P.shape[0]
        d = P.shape[1]
        S = generate_data(P, V, B)
        U, D, Vt = LA.svd(S.T, full_matrices=False)
        sens = np.zeros(n)
        for i in range(n):
            Ui = U[i * (d - 1):(i + 1) * (d - 1),:]     # The relevant chunk of rows from U
            sens[i] = np.linalg.norm(Ui, 'fro')**2
        # Sample coreset
        if C_size == None:
            eps = 0.1
            C_size = int(sum(sens)/eps)
        C_w, C_idx = sampleCoreset(n, sens, C_size)
        return C_w, C_idx, sens


    # Sample a coreset based on the computed sensitivity, and compute new weights for the coreset
    def sampleCoreset(n, sensitivity, sampleSize, weights=None):
        if weights is None:
            weights = np.ones((P.shape[0], 1)).flatten()
        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)
        # The probability of a point prob(p_i) = s(p_i) / t
        probability = sensitivity.flatten() / t
        # The number of points is equivalent to the number of rows in P.
        # initialize new seed
        # np.random.seed()

        # Multinomial Distribution.
        indxs = np.random.choice(n, sampleSize, p=probability.flatten())

        # Compute the frequencies of each sampled item.
        hist = np.histogram(indxs, bins=range(n))[0].flatten()
        indxs = np.nonzero(hist)[0]


        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        # TESTING THE REMOVAL OF
        # hist = np.minimum(np.ones(hist.shape), hist)
        weights = np.asarray(np.multiply(weights[indxs], hist[indxs]), dtype=float).flatten()

        # TESTING WEIGHTS = 1.
        C_weights = np.multiply(weights, 1.0 / (probability[indxs]*sampleSize))
        return C_weights, indxs

    coreset_size = 50
    n = N
    d = D
    B = np.zeros((n,d-1))
    P = np.zeros((n,d))

    # Generate valid V matrices (orthonormal d-1 vectors)
    V = np.zeros((n*(d-1),d))
    for i in range(n):
        R = readdata.read_ortho(i+1)
        curr_V = R[:-1,:]
        V[i*(d-1):(i+1)*(d-1),:] = curr_V
    P = points

    C_w, C_idx, sens = PnP_eps_coreset(P, V, B, coreset_size)

    print("Done creating weighted coreset!")
    return C_idx,C_w
