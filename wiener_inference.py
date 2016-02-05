# Wiener Process Inference

import numpy as np

T = 100
N = 100
MU_SIGMA_X = 1
SIGMA_SIGMA_X = 1
SIGMA_Y = 1

class WienerProcess(object):
  def __init__(self,x,sigma):
    self._x = x
    self._sigma = sigma
  def replicate(self):
    return WienerProcess(self._x,self._sigma)
  def evolve(self,action):
    self._x += sigma*np.random.normal()

def sample_sigma_x():
  return np.random.lognormal(mean=MU_SIGMA_X,sigma=SIGMA_SIGMA_X)

def wiener_inference():
  wp = WienerProcess(sample_sigma_x())
  pf = ParticleFilter(lambda (): WienerProcess(sample_sigma_x()), N=N)
  for i in range(T):
    pf.iterate(None,likelihood)

if __name__ == '__main__':
  wiener_inference()

  
