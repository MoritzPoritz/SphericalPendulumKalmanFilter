import numpy as np
import pandas as pd

def polar_to_kartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = -r * np.cos(theta)
    return [x,y,z]


def write_to_csv(states, pos, Ts):
    pos = np.array(pos)
    Ts = np.array(Ts)
    states = np.array(states)
    print(states.shape, pos.shape, Ts.shape)
    df = pd.DataFrame({
        'theta':states[:,0],
        'phi': states[:,1],
        'dtheta': states[:,2],
        'dphi': states[:,3],
        'position_x': pos[:,0],
        'position_y': pos[:,1],
        'position_z': pos[:,2],
        'timestamps': Ts
        })
    df.to_csv('simulation.txt', sep=',')

