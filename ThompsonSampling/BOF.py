
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt



def BoF(X, codebook=None, k=None, normalize=True, plot=False, ):
    '''
    Bag of Features (Bag of Visual Words)

    Input:
        X: numpy array NxM
        codebook: None if training, the "learnt" codebook otherwise
        k: number of cluseters for k-means. Determines the dimensionality of resulting data
        normalize: if True applies TF-IDF normalization
        plot: if True plots the k-means clustering result

    Returns:
        result: The BoF features as a numpy array (Nxk)
        codebook: The learnt codebook
    '''
    #import code; code.interact(local=dict(globals(), **locals()))
    num_patches = int(len(X[0])/3)

    descriptors = X.reshape(-1, num_patches, 3)
    all_descriptors = X.reshape(-1,3)

    if codebook is None:
                
        #Building the Codebook
        iters = 1
        if k is None:
            dists = []
            min_dist = 1e10
            for kk in range(2,50):
                cb, variance = kmeans(all_descriptors, kk, iters, seed=123)
                dists.append(variance)
                if variance < min_dist:
                    min_dist = variance
                    codebook = cb
            plt.plot(np.arange(2,50), dists, '--o')
            plt.grid()
            plt.show()
        else:
            codebook, _ = kmeans(all_descriptors, k, iters)
            
    #Vector quantization
    visual_words = []
    for i in range(descriptors.shape[0]):
        img_descriptors = descriptors[i,:,:]
        # for each image, map each descriptor to the nearest codebook entry
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
        
    """ if plot:
        vw = np.hstack(visual_words)[:2000]
        cmap = get_cmap(len(np.unique(vw)))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(vw)):
            ax.scatter(all_descriptors[i,0], all_descriptors[i,1], all_descriptors[i,2], color=cmap(vw[i]))
        plt.show() """

    #Frequency count
    frequency_vectors = []
    for img_visual_words in visual_words:
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        #frequency_vectors = np.append(frequency_vectors,img_frequency_vector)
        frequency_vectors.append(img_frequency_vector)
        # stack together in numpy array
    frequency_vectors = np.stack(frequency_vectors)

    if normalize:
        #tf-idf
        N = descriptors.shape[0] 	# N is the number of images, i.e. the size of the dataset
        # df is the number of images that a visual word appears in
        # we calculate it by counting non-zero values as 1 and summing
        df = np.sum(frequency_vectors > 0, axis=0) + 1e-10
        idf = np.log(N/df)
        result = frequency_vectors * idf
    else:
        result = frequency_vectors

    return result, codebook
