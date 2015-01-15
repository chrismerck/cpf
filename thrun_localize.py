import sys
import pygame
from scipy.misc import imread
import numpy as np
from math import sin, cos, pi, exp
from collections import defaultdict
from particle_filter import ParticleFilter
from scipy.stats import norm

pygame.init() 

MAP_FILE = "thrun_office.png"

# load map image
map_img = pygame.image.load(MAP_FILE)
map_rect = map_img.get_rect()
XMAX = map_rect.width
YMAX = map_rect.height

# generate map array
#  where 0 = free, non-zero = wall
map_a = imread(MAP_FILE)
map_a = map_a.reshape(XMAX*YMAX*3)
map_a = map_a[::3]
def map_check_free(x):
  x = np.round(x)
  if (x[0] < 0 or x[0] >= XMAX or x[1] < 0 or x[1] >= YMAX):
    return False
  return map_a[x[0]+x[1]*XMAX] == 0

#create the screen to be size of map
window = pygame.display.set_mode((XMAX,YMAX))

SIGMA_S = 0.2
SIGMA_H = 0.005 * pi * 2;
SIGMA_RANGE = 5
N_RANGE_FINDERS = 7 #8
RANGE_FINDER_MAX_RANGE = 300.0

def raylen(x):
  return np.sqrt(x.dot(x))

def raytrace(x0,dx,slide=False):
  # given initial position x0 and a displacement dx,
  #   return the point x1 furthest along the line (x0,x0+dx)
  #   before a wall is hit
  _x = np.array(x0)
  dx = np.array(dx)
  dxlen = raylen(dx)
  ddx = dx / max(1,dxlen)
  for i in range(int(dxlen)):
    if not map_check_free(_x + ddx):
      break
    _x += ddx
  return _x

def unitcircle(phi):
  return np.array([cos(phi),sin(phi)])

class Robot(object):
  def __init__(self,x0,h0,perturbance=0):
    # create a new robot with specified initial state
    self._x = np.array(map(float,x0))
    self._h = h0
    self._perturbance = perturbance

  def replicate(self):
    # return a new copy of our state
    return Robot(self._x,self._h,self._perturbance).perturb()
  
  def perturb(self):
    # perturb if required
    self._x += self._perturbance*np.random.normal(size=2)
    return self

  def evolve(self,action):
    # evolve ds steps forward and change angle by dh
    #  with a stochastic component (based on SIGMA_X and SIGMA_H)
    #  but stop when we hit a wall
    (ds,dh) = action
    """ds = ds + SIGMA_S*self._perturbance*np.random.normal()
    dh = dh + SIGMA_H*self._perturbance*np.random.normal()"""
    #
    #  compute new heading
    self._h += dh + np.random.normal()*SIGMA_H
    # compute displacement
    dx = ((ds + np.random.normal()*SIGMA_S)*unitcircle(self._h))
    # move towards goal one step at a time,
    #  refusing to enter wall
    self._x = raytrace(self._x,dx)

  def sense(self,surf=None):
    # compute ranges detected by range sensors
    ranges = []
    range_xs = []
    for i in range(N_RANGE_FINDERS):
      phi = self._h + i*pi*2/float(N_RANGE_FINDERS)
      dx = RANGE_FINDER_MAX_RANGE*unitcircle(phi)
      range_xs.append(raytrace(self._x,dx))
      ranges.append(raylen(range_xs[-1]) + SIGMA_RANGE*np.random.normal())
    # optionally draw the range finders
    if surf != None:
      for i in range(N_RANGE_FINDERS):
        pygame.draw.line(surf,(0,255,0),self._x,range_xs[i])
    return np.array(ranges)

  def draw(self,surf,color=(255,0,0)):
    arrow_len = 10
    x1 = self._x
    x2 = self._x + arrow_len*np.array([cos(self._h),sin(self._h)])
    x3 = self._x + 0.8*arrow_len*np.array([cos(self._h+pi/8),sin(self._h+pi/8)])
    x4 = self._x + 0.8*arrow_len*np.array([cos(self._h-pi/8),sin(self._h-pi/8)])
    pygame.draw.line(surf,color,x1,x2)
    pygame.draw.line(surf,color,x2,x3)
    pygame.draw.line(surf,color,x2,x4)

def particleGeneratorGenerator(perturbance):
  def _():
    # uniformly distributed robots around the free space of the map
    while True:
      x = [np.random.random()*float(XMAX),np.random.random()*float(YMAX)]
      if map_check_free(x):
        return Robot(x,np.random.random()*pi*2, perturbance)
  return _

class Game(object):
  def __init__(self):
    self._robot = Robot((XMAX/2+100,YMAX/2-25),0)
    #self._robot = particleGeneratorGenerator(0)()
    self._keydown = defaultdict(bool)
    self.reset_pf()
  def reset_pf(self):
    self._pf = ParticleFilter(particleGeneratorGenerator(5),N=500,regenProb=0.2)
  def draw(self,surf):
    #self._robot.sense(surf)
    #self._robot.draw(surf,(255,255,0))
    for i in range(len(self._pf._particles)):
      p = self._pf._particles[i]
      p.draw(surf,color=(255,0,0))
  def evolve(self):
    pause = True
    ds = 0
    dh = 0

    if self._keydown[pygame.K_x]:
      pause = False

    if self._keydown[pygame.K_r]:
      self.reset_pf()
      pause = False

    if self._keydown[pygame.K_UP]:
      pause = False
      ds = 2
    elif self._keydown[pygame.K_DOWN]:
      pause = False
      ds = -2
    if self._keydown[pygame.K_RIGHT]:
      pause = False
      dh = pi/32
    elif self._keydown[pygame.K_LEFT]:
      pause = False
      dh = -pi/32
    if not pause:
      action = (ds,dh)
      self._robot.evolve(action)
      ranges = self._robot.sense()
      def likelihood(p):
        ranges_prime = p.sense()
        ranges_diff = np.abs(ranges_prime - ranges)
        rv = 1.0
        for i in range(len(ranges_diff)):
          #rv *= exp(-(ranges_diff[i]/SIGMA_RANGE)**2)*0.5 + 0.5
          if (ranges_diff[i] / SIGMA_RANGE) > 2:
            rv *= 0.3
          elif (ranges_diff[i] / SIGMA_RANGE) > 2:
            rv *= 0.5
          elif (ranges_diff[i] / SIGMA_RANGE) > 1:
            rv *= 0.9
        return rv

      if not self._keydown[pygame.K_h]:
        #self._pf.iterate(action,likelihood)
        self._pf._evolve(action)
        if self._keydown[pygame.K_x]:
          self._pf._update(likelihood)

  def handle_event(self,event):
    if event.type == pygame.KEYDOWN:
      self._keydown[event.key] = True
    elif event.type == pygame.KEYUP:
      self._keydown[event.key] = False

g = Game()

def draw():

  # draw map
  window.blit(map_img, map_img.get_rect())

  g.evolve()
  g.draw(window)

  #draw it to the screen
  pygame.display.flip() 

#input handling (somewhat boilerplate code):
mouse_over = True
while True: 
  for event in pygame.event.get(): 
    if event.type == pygame.QUIT: 
      sys.exit(0) 
    elif event.type == pygame.ACTIVEEVENT:
      mouse_over = (event.gain==1)
    elif event.type == pygame.MOUSEMOTION:
      if mouse_over:
        (x,y) = event.pos
    else: 
      pass
      #print event
    g.handle_event(event)

  draw()

