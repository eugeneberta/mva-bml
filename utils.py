import torch
import numpy as np
import seaborn as sns

from torch import nn
from torch.distributions import MultivariateNormal


class GaussianParticle(nn.Module):
    """A class to handle simple Gaussian distributions and Gaussian particles when using the boosting procedure.
    We cannot simply re-use torch.distribution.MultivariateNormal as we also need to take care of the mixture
    parameter lambda.
    """
    def __init__(
            self,
            loc:torch.FloatTensor=torch.zeros(2),
            scale_tril:torch.FloatTensor=torch.eye(2),
            trainable:bool=True
        ):
        """
        Args:
            loc (torch.FloatTensor, optional): Mean at initialization. Defaults to torch.zeros(2).
            scale_tril (torch.FloatTensor, optional): L matrix in the Cholesky decomposition of the covariance matrix
            S = L.T@L. Defaults to torch.eye(2).
            trainable (bool, optional): Wether the parameters of the distribution can be changed by the training
            procedure. Defaults to True.
        """
        super().__init__()
        self.dim = len(loc)
        self.trainable = trainable
        # Parameters of the distribution:
        self.loc = nn.parameter.Parameter(loc, requires_grad=self.trainable)
        self.scale_tril = nn.parameter.Parameter(scale_tril, requires_grad=self.trainable)
        # Internal parameter only, call lmbda() method for access.
        self._lmbda = nn.parameter.Parameter(torch.tensor(0.), requires_grad=self.trainable)
        # We use a PyTorch distribution to compute the density.
        self.mvn = MultivariateNormal(loc=self.loc, scale_tril=self.scale_tril)

    def lmbda(self):
        # Trick to implement the [0,1] constraint on lambda. This way we can optimize lambda without constraints
        # and still get a parameter value in [0,1].
        return torch.sigmoid(self._lmbda)

    def log_prob(self, X):
        return self.mvn.log_prob(X)

    def sample(self, size=[1000]):
        # Reparametrization trick:
        std_normal = MultivariateNormal(loc=torch.zeros(self.dim), scale_tril=torch.eye(self.dim))
        samples = std_normal.sample(size)
        samples = self.loc + samples @ self.scale_tril.T
        return samples


class GaussianMixture(nn.Module):
    """A class to build Gaussian Mixture Models (GMM) with several GaussianParticle.
    """
    def __init__(self, init:GaussianParticle):
        """The GMM is initialized with a single Gaussian Particle and then populated using the add_particle method.

        Args:
            init (GaussianParticle): GMM Initialization, first particle of the mixture.
        """
        super().__init__()
        # GMM Parameters:
        self.weights = torch.tensor([1])
        self.particles = [init]

    def add_particle(self, particle:GaussianParticle):
        """Adding a particle to the mixture.

        Args:
            particle (GaussianParticle): New particle.
        """
        # Rescaling current weights:
        self.weights = self.weights*(1-particle.lmbda())
        # Updating weights list:
        self.weights = torch.cat([self.weights, torch.tensor([particle.lmbda()])])
        # Updating particles list:
        self.particles.append(particle)

    def log_prob(self, X:torch.FloatTensor):
        # log_prob method to be coherent with the PyTorch.distributions API
        prob = torch.zeros(len(X))
        for i in range(len(self.particles)):
            prob += self.weights[i]*torch.exp(self.particles[i].log_prob(X))
        return torch.log(prob)

    def sample(self, size=[1000]):
        # sample method to be coherent with the PyTorch.distributions API
        samples = []
        # Sampling in a hierarchical model. We first pick the particle to sample from for each sample,
        # using the weights as a probability distribution:
        indexes = np.random.choice(len(self.particles), size=size[0], p=self.weights.detach().numpy())
        # Number of samples to take in each particle:
        bincount = np.bincount(indexes)
        for i in range(len(bincount)):
            samples.append(self.particles[i].sample(size=[bincount[i]]))
        return torch.cat(samples)


def MC_RKL(q, p, n_samples=1000):
    """Computes the Reverse Kullback-Leibler divergence KL(q || p) using a Monte Carlo estimate.
    /!\ When optimizing the model, using this loss comes to optimizing the distribution used for sampling.
    This results in a high variance estimate and optimization procedure.

    Args:
        q (torch.distribution): Model distribution.
        p (torch.distribution): Target distribution.
        n_samples (int, optional): Number of samples to use for the Monte Carlo estimate. Defaults to 1000.

    Returns:
        float: Reverse KL divergence estimate.
    """
    samples = q.sample([n_samples])
    return torch.mean(q.log_prob(samples) - p.log_prob(samples))

def MC_FKL(q, p, nsamples=1000):
    """Computes the Forward Kullback-Leibler divergence KL(p || q) using a Self-Normalized
    Importance Sampling (SNIS) estimate.
    /!\ When optimizing the model, using this loss comes to optimizing the distribution used for sampling.
    This results in a high variance estimate and optimization procedure.

    Args:
        q (torch.distribution): Model distribution.
        p (torch.distribution): Target distribution.
        nsamples (int, optional): Number of samples to use for the SNIS estimate. Defaults to 1000.

    Returns:
        float: Forward KL divergence estimate.
    """
    samples = q.sample([nsamples])
    log_ratio = p.log_prob(samples) - q.log_prob(samples)
    r = torch.exp(log_ratio)
    w = r / r.sum()
    return torch.dot(w, log_ratio)

def Boost_FKL(p, q, f, nsamples=1000):
    """Computes the Forward Kullback-Leibler divergence KL(p || q) using a SNIS estimate by sampling in
    an initialization distribution f.
    Implementation of the method described in the paper "Variational refinement for importance sampling
    using the forward kullback-leibler divergence, 2021" to reduce the variance of the SNIS estimate in
    MC_FKL.

    Args:
        p (torch.distribution): Target distribution.
        q (torch.distribution): Model distribution.
        f (torch.distribution): Initialization distribution.
        nsamples (int, optional): Number of samples to use for the SNIS estimate. Defaults to 1000.

    Returns:
        float: Forward KL divergence estimate.
    """
    samples = q.sample([nsamples])

    ps = p.log_prob(samples)
    qs = q.log_prob(samples)
    fs = f.log_prob(samples)

    l = f.lmbda()

    r = torch.exp(ps-qs)
    w = r / r.sum()

    return torch.dot(w, ps - torch.log(l*torch.exp(fs) + (1-l)*torch.exp(qs)))

def plot_distribs(dict:dict, nsamples:int=1000):
    """Function to plot several distributions on the same seaborn KDE plot.

    Args:
        dict (dict): Dictionnary of distributions to plot with the structure: {'distribution label': distribution}
        nsamples (int, optional): Number of samples to use for the KDE plot. Defaults to 1000.
    """
    x, y, hue = [], [], []
    for key in dict.keys():
        distr = dict[key]
        samples = distr.sample([nsamples]).detach().numpy().squeeze()
        x.append(samples[:,0])
        y.append(samples[:,1])
        hue.append(nsamples*[key])
    x = np.concatenate(x)
    y = np.concatenate(y)
    hue = np.concatenate(hue)

    sns.set_style('white')
    sns.displot(
        x=x,
        y=y,
        hue=hue,
        kind='kde',
        palette=sns.color_palette("hls", len(dict)),
        height=3
    )