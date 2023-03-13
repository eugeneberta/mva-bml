import torch
from torch import nn
from torch.distributions import MultivariateNormal

import numpy as np
import seaborn as sns


class GaussianParticle(nn.Module):    
    def __init__(
            self,
            loc=torch.zeros(2),
            scale_tril=torch.eye(2),
            trainable=True
        ):
        super().__init__()
        self.dim = len(loc)
        self.trainable = trainable
        self.loc = nn.parameter.Parameter(loc, requires_grad=self.trainable)
        self.scale_tril = nn.parameter.Parameter(scale_tril, requires_grad=self.trainable)
        self._lmbda = nn.parameter.Parameter(torch.tensor(0.), requires_grad=self.trainable)
        
        self.mvn = MultivariateNormal(loc=self.loc, scale_tril=self.scale_tril)

    def lmbda(self):
        # Trick to implement the [0,1] constraint on lambda:
        return torch.sigmoid(self._lmbda)

    def log_prob(self, X):
        return self.mvn.log_prob(X)

    def sample(self, size=[1000]):
        # if self.trainable:
        #     # Reparametrization trick:
        #     std_normal = MultivariateNormal(loc=torch.zeros(self.dim), scale_tril=torch.eye(self.dim))
        #     samples = std_normal.sample(size)
        #     samples = self.loc + samples @ self.scale_tril.T
        #     return samples
        return self.mvn.rsample(size)


class GaussianMixture(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.weights = torch.tensor([1])
        self.particles = [init]

    def add_particle(self, particle):
        # updating weights list:
        self.weights = self.weights*(1-particle.lmbda())
        self.weights = torch.cat([self.weights, torch.tensor([particle.lmbda()])])
        # updating particles list:
        self.particles.append(particle)

    def log_prob(self, X):
        # log_prob to be coherent with PyTorch distributions API
        prob = torch.zeros(len(X))
        for i in range(len(self.particles)):
            prob += self.weights[i]*torch.exp(self.particles[i].log_prob(X))
        return torch.log(prob)

    def sample(self, size=[1000]):
        samples = []
        # sampling in a hierarchical model:
        indexes = np.random.choice(len(self.particles), size=size[0], p=self.weights.detach().numpy())
        bincount = np.bincount(indexes)
        for i in range(len(bincount)):
            samples.append(self.particles[i].sample(size=[bincount[i]]))
        return torch.cat(samples)


def RKL(q, p, n_samples=1000):
    samples = q.sample([n_samples])
    return torch.mean(q.log_prob(samples) - p.log_prob(samples))

def varFKL(p, q, nsamples=1000):
    samples = q.sample([nsamples])

    ps = p.log_prob(samples)
    qs = q.log_prob(samples)

    r = torch.exp(ps-qs)
    w = r / r.sum()

    return torch.dot(w, ps - qs)

def FKL(p, q, f, nsamples=1000):
    samples = q.sample([nsamples])

    ps = p.log_prob(samples)
    qs = q.log_prob(samples)
    fs = f.log_prob(samples)

    l = f.lmbda()

    r = torch.exp(ps-qs)
    w = r / r.sum()

    return torch.dot(w, ps - torch.log(l*torch.exp(fs) + (1-l)*torch.exp(qs)))

def plot_distribs(dict, nsamples=1000):
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