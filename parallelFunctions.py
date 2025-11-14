# import modules
from sca.models import fit_sca, weighted_pca
from sca.util import get_sample_weights
import numpy as np


''' 
simple function that just runs SCA when provided...
R_est: number of SCA dimensions
orthLam: non-orthogonality penalty
sparseLam: non-sparsity pendalty 
monkName: Alex or Balboa

This function just exists because of a bug that exists when trying to parallelize stuff in notebooks 

'''
def runSCA(R_est,orthLam, sparseLam,monkName):

    # load data
    saveDir = '/Users/andrew/Documents/Projects/Churchland/Sparsity/data/reaching_orthSparsitySweep/'
    data = np.load(saveDir + monkName + '_dataForSCAPool.npy',allow_pickle = True)
    data = data.item()
    fit_data = data['fit_data']
    trainMask = data['trainMask']
    sample_weights = data['sample_weights']
    hardOrthFlag = data['hardOrthFlag']

    # fit sca model
    model,sca_latent, x_pred,losses=fit_sca(X=fit_data[trainMask,:],sample_weight=sample_weights[trainMask],
                                            R=R_est, orth=hardOrthFlag, lam_orthog=orthLam, lam_sparse=sparseLam)

    # pop off latents and reconstructions
    sca_latent = sca_latent.detach().numpy()
    x_pred     = x_pred.detach().numpy()

    return sca_latent, x_pred, R_est, orthLam, sparseLam

''' 
function that samples neurons (with replacement) and runs SCA and weighted PCA using the default parameters for both

as above, the only reason this function exists is as a workaround for a known bug. 

inputs: 
R_est: number of dimensions
X: CT x N matrix of trial averaged rates 
trainMask: CT x 1 mask of times we want to use for analyses

note: X should already be pre-processed (downsampled in time, soft-normalized, mean-subtracted) 
'''
def bootstrapNeurons_SCA_PCA(R_est,X,trainMask):

    ### redraw our neurons ###

    # number of neurons
    numN = X.shape[1]

    # random index of neurons
    randIdx = np.random.randint(0, numN, size=numN)

    # grab our random neurons
    X_sample = X[:,randIdx]


    ### calculate sample weights ###
    sw = get_sample_weights(X_sample)

    ### run sca ###
    model, sca_latent, x_pred, losses = fit_sca(X=X_sample[trainMask, :], sample_weight=sw[trainMask],
                                                R=R_est, orth=False)

    # project all the data into the sca space
    sca_latents = X_sample @ model.fc1.weight.detach().numpy().T


    ### run pca ###
    U_est, V_est = weighted_pca(X_sample[trainMask, :], R_est, sw[trainMask])

    # project all the data
    pca_latents = X_sample@U_est

    # return latents
    return sca_latents, pca_latents