import numpy as np

def ObstacleHit(cost_map, position):
    y = position[0]
    x = position[1]
    if cost_map[int(np.floor(x)),int(np.floor(y))] != 0 and cost_map[int(np.ceil(x)),int(np.ceil(y))] != 0:
        return True
    return False