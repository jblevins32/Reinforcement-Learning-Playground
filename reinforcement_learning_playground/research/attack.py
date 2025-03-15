import numpy as np

class Attacker():
    def __init__(self, attack_type):
        self.attack_type = attack_type

    def action_attack(self, action):

        if self.attack_type == "reflection":
            Su = -1
        elif self.attack_type == "permutation":
            Su = np.array([[0 , 1], [1, 0]])
        return np.dot(Su,action)

    def obs_attack(self,obs, obs_prev):

        if self.attack_type == "reflection":
            Sx = -1
            dx = obs_prev - Sx*obs_prev
        elif self.attack_type == "permutation":
            Sx = np.array([[0 , 1], [1, 0]])
            dx = obs_prev - np.dot(Sx,obs_prev)
        return np.dot(Sx,obs) + dx