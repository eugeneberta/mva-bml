import numpy as np
import seaborn as sns

# def plot_distribs(lst, lst_colors):
#     delta = 0.1
#     delta_x = 2
#     delta_y = 2
#     for i in range(d):
#         d = lst[i]
#         c = lst_colors[i]
#         mu = d.loc.detach().numpy()
#         x = np.arange(mu[0]-delta_x, mu[0]+delta_x, delta)
#         y = np.arange(mu[1]-delta_y, mu[1]+delta_y, delta)
#         X, Y = np.meshgrid(x, y)
#         pos = np.empty(X.shape + (2,))
#         pos[:, :, 0] = X; pos[:, :, 1] = Y
#         pos = torch.tensor(pos)
#         Z = torch.exp(d.log_prob(pos)).detach().numpy()
#         plt.contour(X, Y, Z, levels=2, colors=c)
#     plt.show()

def plot_distribs(lst, nsamples=1000):
    d = len(lst)
    x = np.zeros(d*nsamples)
    y = np.zeros(d*nsamples)
    hue = np.zeros(d*nsamples)

    for i in range(d):
        distr = lst[i]
        samples = distr.sample([nsamples]).detach().numpy().squeeze()
        x[nsamples*i:nsamples*i+nsamples] = samples[:,0]
        y[nsamples*i:nsamples*i+nsamples] = samples[:,1]
        hue[nsamples*i:nsamples*i+nsamples] = np.ones(nsamples)*(i+1)

    sns.displot(
        x=x,
        y=y,
        hue=hue,
        kind='kde',
        palette=sns.color_palette("hls", d)
    )