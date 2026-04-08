import numpy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from create_latent_ode_model import create_LatentODE_model
from torch.distributions.normal import Normal

#import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import matplotlib.cm as cm

total_latents = np.load('latents.npy')  ## Load the latent arrays
divergence = np.load('divergence.npy')  ## Load the divergence arrays



total_latents = total_latents.reshape(-1, total_latents.shape[-1])
divergence = divergence.reshape(-1, 1)
print("Divergence: ", divergence.shape)


time_start = time.time()
try:
    tsne_result = np.load('tsne_rnn_out8_result12.npy')
    print("Loaded tsne results")
except:
    print("TSNE Not Loaded")
    tsne = TSNE(n_components=2, perplexity=500)
    tsne_result = tsne.fit_transform(total_latents)
print("Tsne: ", tsne_result.shape)

x_meshgrid = np.linspace(3/2*min(tsne_result[:,0]), 3/2*max(tsne_result[:,0]), 100)
y_meshgrid = np.linspace(3/2*min(tsne_result[:,1]), 3/2*max(tsne_result[:,1]), 100)
X,Y = np.meshgrid(x_meshgrid, y_meshgrid)
print("X: ", X.shape)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

def gaussian_kernel(distance, sigma=1):
    return np.exp(-0.5 * (distance / sigma)**2) + 1e-4


distances = cdist(tsne_result, np.column_stack((X.ravel(), Y.ravel())))
print("Distance: ", distances.shape)
weights = gaussian_kernel(distances) # Compute weights based on distances
weights = weights.T
print("Weights: ", weights.shape)
weighted_divergence = np.matmul(weights, divergence).squeeze()
sum_weights = np.sum(weights, axis=1)
print("Weighted Div: ", weighted_divergence.shape)
print("Sum weights: ", sum_weights.shape)
weighted_divergence = np.divide(weighted_divergence, sum_weights)

divergence_interp = weighted_divergence.reshape(X.shape)
plt.figure(figsize=(10, 8))

plt.contourf(X,Y, divergence_interp, cmap='viridis') 

## PLOT TSNE LATENTS
tsne_result = tsne_result.reshape(all_latents.shape[0], all_latents.shape[1], tsne_result.shape[-1])
# colors = cm.viridis(np.linspace(0, 1, len(tsne_result)))
for i, result in enumerate(tsne_result):

    plt.plot(result[:, 0], 
                    result[:, 1],
                    alpha=0.7
                    # color = colors[i]
                    )
    # Mark the starting point
    plt.scatter(result[0, 0], 
                result[0, 1], 
                color='green', 
                label=f'Start {i+1}' if i == 0 else "", zorder=5)

    # Mark the ending point
    plt.scatter(result[-1, 0], 
                result[-1, 1], 
                color='red', 
                label=f'End {i+1}' if i == 0 else "", zorder=5)

plt.legend(loc='lower left', bbox_to_anchor=(1, 1))

plt.colorbar(label='Divergence')


plt.title('t-SNE Divergence')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('tsne.png')