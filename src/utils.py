import numpy as np
import pandas as pd
import pathlib


def polar_to_kartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = -r * np.cos(theta)
    return [x,y,z]


def write_to_csv(states, pos, Ts, name):
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
        'timestep': Ts
        })
    df.to_csv('data/'+name+'.csv', sep=',')


def read_from_csv(path):
    df = pd.read_csv('data/'+ path, skipinitialspace=True)
    return df

def simulation_data_to_array(df): 
    df_state = df[['theta','phi', 'dtheta', 'dphi']]
    df_positions = df[['position_x','position_y', 'position_z']]
    df_ts = df['timestep']
    state_array = df_state.to_numpy().tolist()
    positions_array = df_positions.to_numpy().tolist()
    ts_array = df_ts.to_numpy().tolist()

    return state_array, positions_array, ts_array
