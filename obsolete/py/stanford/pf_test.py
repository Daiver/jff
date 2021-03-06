# In this exercise, please run your previous code twice.
# Please only modify the indicated area below!

from math import *
import random
from particle_filter import particleFilter
import cv2

import numpy as np
landmarks  = [ [280.0, 280.0], [30.0, 120.0], [120., 90.]]

world_size = 400.0

class robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 0.0;
    
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError, 'X coordinate out of bound'
        if new_y < 0 or new_y >= world_size:
            raise ValueError, 'Y coordinate out of bound'
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    
    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError, 'Robot cant move backwards'         
        
        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size    # cyclic truncate
        y %= world_size
        
        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    
    def measurement_prob(self, measurement):
        
        # calculates how likely a measurement should be
        
        prob = 1.0;
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
    
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))



def eval(r, p):
    sum = 0.0;
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err
    return sum / float(len(p))

def vizAll(myrobot, parts, landmarks):
    img = np.zeros((world_size, world_size, 3))
    img[:] = 255
    cv2.circle(img, (int(myrobot.x), int(myrobot.y)), 3, (5, 0, 0))
    for x in parts:
        cv2.circle(img, (int(x.x), int(x.y)), 1, (0, 0, 255))

    for x in landmarks:
        cv2.line(img, (int(x[0]), int(x[1])), (int(myrobot.x), int(myrobot.y)), (255, 0, 0))
        cv2.circle(img, (int(x[0]), int(x[1])), 3, (0, 255, 0))
    return img

#myrobot = robot()
#myrobot.set(30.0, 50.0, pi/2)
#myrobot = myrobot.move(-pi/2, 15.0)
#print myrobot.sense()
#myrobot = myrobot.move(-pi/2, 10.0)
#print myrobot.sense()

if __name__ == '__main__':
    myrobot = robot()
    myrobot = myrobot.move(0.1, 5.0)
    myrobot.set_noise(5.0, 0.1, 5.0)
    Z = myrobot.sense()
    N = 1000
    T = 10 #Leave this as 10 for grading purposes.

    p = []
    for i in range(N):
        r = robot()
        r.set_noise(0.05, 0.05, 5.0)
        p.append(r)

    Z = myrobot.sense()
    p = particleFilter(
        lambda x: x.measurement_prob(Z),
        lambda p: [x.move(0.1*random.random(), 0.5*random.random()) for x in p],
        p,
        T
    )
    print eval(myrobot, p)
    print repr(myrobot)
    print repr(p[0])
    print repr(p[-1])


