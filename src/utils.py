import numpy as np
import pandas as pd
from datetime import datetime

def write_evaluation_to_csv(index, mse, mse_db, error_in_place, occ_start, occ_end,occ_dur, Q_var, meaned_kalman_var, crashed, crashed_at): 
    df = pd.DataFrame({
        'index': index, 
        'Mean Squared Error':mse, 
        'Mean Squared Error dB':mse_db, 
        'Error in Filter Std':error_in_place,
        'Occlusion Start': occ_start, 
        'Occlusion End': occ_end,
        'Occlusion Duration': occ_dur, 
        'System Noise': Q_var, 
        'Meaned Filter Variance': meaned_kalman_var, 
        'Filter Crashed': crashed, 
        'Crashed At': crashed_at
        })
    df.to_csv('data/Evaluation_'+str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+'.csv', sep=',')

def polar_to_kartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = -r * np.cos(theta)
    return [x,y,z]

def cartesian_to_polar(x,y,z): 
    theta = np.arctan(y/x)
    phi = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    return theta, phi

def write_to_csv(states, pos, Ts, name, vars):
    
    if (len(vars) >0): 
        print("HALLO")
        pos = np.array(pos)
        Ts = np.array(Ts)
        states = np.array(states)
        vars = np.array(vars)
        print(states.shape, pos.shape, vars.shape,Ts.shape)
        df = pd.DataFrame({
            'theta':states[:,0],
            'phi': states[:,1],
            'dtheta': states[:,2],
            'dphi': states[:,3],
            'position_x': pos[:,0],
            'position_y': pos[:,1],
            'position_z': pos[:,2],
            'var_theta': vars[:,0],
            'var_phi': vars[:,1],
            'var_dtheta': vars[:,2],
            'var_dphi': vars[:,3], 
            'timestep': Ts
            })
    else: 
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
