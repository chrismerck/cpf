# particle_filter.py
# Modular Particle Filter Implementation
# (c) 2015 Chris Merck

import numpy as np
import scipy
from multiprocessing import Pool

p = Pool(16)
def fast_map(f, array):
  #return p.map(f,array)
  return map(f,array)

class ParticleFilter(object):
  def __init__(self,particleGenerator,N,regenProb=0):
    self._particles = []
    self._t = 0
    self._N = N
    self._regenProb = regenProb
    self._particleGenerator = particleGenerator
    for i in range(self._N):
      self._particles.append(particleGenerator())
  def iterate(self,action,likelihood):
    self._evolve(action)
    self._update(likelihood)
  def _evolve(self,action):
    # prediction step of GSS93 
    for i in range(self._N):
      p = self._particles[i]
      p.evolve(action)
  def _update(self,likelihood):
    # regenerate new particles sometimes
    for i in range(self._N):
      if (np.random.random() < self._regenProb):
        self._particles[i] = self._particleGenerator()
    # compute particle likelihoods
    q = np.array(fast_map(likelihood, self._particles))
    # convert PMF->CMF
    q = np.cumsum(q)
    # normalize
    q /= q[-1]
    # resample
    r = np.random.random(self._N)
    js = np.searchsorted(q,r)
    new_particles = []
    for j in js:
      p = self._particles[j]
      new_particles.append(p.replicate())
    self._particles = new_particles

if __name__=="__main__":
  pf = ParticleFilter()

