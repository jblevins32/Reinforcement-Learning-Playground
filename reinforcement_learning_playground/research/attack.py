class Attacker():
    def __init__(self):
        self.attack_type = 1

    def action_attack(self, action):
        Su = 0.1
        return Su * action

    def obs_attack(self,obs, obs_prev):
        Sx = 1/0.1
        dx = obs_prev - Sx*obs_prev
        return Sx*obs + dx