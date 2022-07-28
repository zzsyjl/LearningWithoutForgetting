import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_fig(x, y, file_name):
    ckpt_dir = "./images/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    x_tsne =TSNE(n_components=2, random_state=10).fit_transform(x)

    plt.figure()
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
    plt.savefig(os.path.join(ckpt_dir, file_name), bbox_inches="tight")
