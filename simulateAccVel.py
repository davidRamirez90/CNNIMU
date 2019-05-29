import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import env
import pdb


sequrl = "/vol/corpora/har/MoCap/recordings_2019/03_Markers_Exports/{}"


def read_data(path):
        '''
        Reads CSV file from path
        @:param File path
        '''
        data = pd.read_csv(path, skiprows=2)
        data = data.dropna(axis=1, how='all')
        data = data.iloc[2:, 2:]
        data = data.to_numpy()

        return data

def coords_to_channels(data):

    res = np.array([
        np.copy(data[:, ::3]),
        np.copy(data[:, 1::3]),
        np.copy(data[:, 2::3])
    ])
    print(res)
    return res

def diff(base, dt):
    dbase = list()
    for i, pos in enumerate(base):
        if i == 0:
            prevpos = pos
        else:
            dif = (pos - prevpos) / dt
            dbase.append(dif)
            prevpos = pos
    return np.asarray(dbase)
    
def plot(data, t):
    (pos, vel, acc) = data
    fig, axis = plt.subplots(3, 1)
    axis[0].plot(t, pos)
    axis[0].set_xlabel('Time')
    axis[0].set_ylabel('pos')
    axis[1].plot(t, vel)
    axis[1].set_xlabel('Velocity')
    axis[1].set_ylabel('vel')
    axis[2].plot(t, acc)
    axis[2].set_xlabel('Acceleration')
    axis[2].set_ylabel('acc')
    plt.show()


def calculate(data):
    dt = 1/120
    seconds = data.size / 120
    time = np.arange(0, seconds, dt)
    pos = data
    vel = diff(pos, dt)
    acc = diff(vel, dt)

    plot((np.resize(pos, acc.shape), 
          np.resize(vel, acc.shape), 
          acc), 
         np.resize(time, acc.shape))



if __name__ == "__main__":

    print("Starting program execution")

    d = read_data(sequrl.format('Subject01_01.csv'))
    d2 = coords_to_channels(d)
    d = d[:,0].astype(float)
    
    calculate(d)

    # x = np.arange(0, 10, (1/120))
    # y = np.sin(x)
    # calculate(y)


