import sys, math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
SPEED = 0.29  # Will fall forward on higher speed

class BipedalWalkerExpert:
    def __init__(self):
        self.state = STAY_ON_ONE_LEG
        self.moving_leg = 0
        self.supporting_leg = 1 - self.moving_leg
        self.SUPPORT_KNEE_ANGLE = +0.1
        self.supporting_knee_angle = self.SUPPORT_KNEE_ANGLE

    def get_next_action(self, s):
        a = np.array([0.0, 0.0, 0.0, 0.0])

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*self.moving_leg
        supporting_s_base = 4 + 5*self.supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if self.state==STAY_ON_ONE_LEG:
            hip_targ[self.moving_leg]  = 1.1
            knee_targ[self.moving_leg] = -0.6
            self.supporting_knee_angle += 0.03
            if s[2] > SPEED: self.supporting_knee_angle += 0.03
            self.supporting_knee_angle = min( self.supporting_knee_angle, self.SUPPORT_KNEE_ANGLE )
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                self.state = PUT_OTHER_DOWN
        if self.state==PUT_OTHER_DOWN:
            hip_targ[self.moving_leg]  = +0.1
            knee_targ[self.moving_leg] = self.SUPPORT_KNEE_ANGLE
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[moving_s_base+4]:
                self.state = PUSH_OFF
                self.supporting_knee_angle = min( s[moving_s_base+2], self.SUPPORT_KNEE_ANGLE )
        if self.state==PUSH_OFF:
            knee_targ[self.moving_leg] = self.supporting_knee_angle
            knee_targ[self.supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                self.state = STAY_ON_ONE_LEG
                self.moving_leg = 1 - self.moving_leg
                self.supporting_leg = 1 - self.moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        return a
