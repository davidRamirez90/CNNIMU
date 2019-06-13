from __future__ import print_function
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import logging
import pickle
from sliding_window import sliding_window
import time
import glob
import tqdm
import env
import pdb
import re

# LOGGING CONFIGURATION
# logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
# logging.basicConfig(filename='debug.log', level=logging.DEBUG, format=logging_format)
# logger = logging.getLogger('CNN network')


# Added data to env.py
# MOCAP MARKERS
# markers_url = "/vol/corpora/har/MoCap/recordings_2019/03_Markers_Exports/Subject{}_{}.csv"
# marker_window_url = "/data/dramirez/dataset_wsize={}_wstride={}/{}"
# marker_models_url = "/data/dramirez/marker_models"


class WindowGenerator:

    def __init__(self, win_size=100, win_stride=1):
        self.win_size = win_size
        self.win_stride = win_stride
        self.dataset_dir = env.dataset_url
        self.save_dataset_dir = env.window_url
        self.markerset_dir = env.markers_url
        self.save_marker_dataset_dir = env.marker_window_url


    def read_data(self, path):
        '''
        Reads CSV file from path
        @:param File path
        '''
        data = pd.read_csv(path)
        data = data.dropna(axis=1, how='all')
        data = data.iloc[:, 1:]
        data = data.to_numpy()

        return data

    def read_data_markers(self, path):
        '''
        Reads CSV file from path
        @:param File path
        '''
        data = pd.read_csv(path, skiprows=2)
        data = data.dropna(axis=1, how='all')
        data = data.iloc[2:-2, 2:]
        data = data.to_numpy()

        return data

    def coords_to_channels(self, data):

        res = np.array([
            np.copy(data[:, ::3]),
            np.copy(data[:, 1::3]),
            np.copy(data[:, 2::3])
        ])
        return res

    def getMostCommonClass(self, window):
        '''
        Looks for most common class within the provided window
        @:param windows: Window to be analyzed
        @:return class: Most common class found on window
        '''
        if window.shape.__len__() > 1:
            classes = window[:, 0]
        else:
            classes = window
        mid = classes[round(classes.size / 2)]

        return mid


    def removeClass(self, data, classN):
        '''
        :param data: Data to remove class from
        :param classN: Class number to remove from data
        :return: filtArray: Filtered Array
        '''
        indices = data[:, 0] != classN
        filtArray = data[indices, :]

        return filtArray


    def normalizeData(self, data, haslabels=True):
        '''
        :param data: Input data to normalize
        :return: Normalized data, 0 mean, Unit variance, column wise
        '''
        if haslabels:
            labels = np.transpose(np.asmatrix(data[:, 0]))
            content = preprocessing.scale(data[:, 1:])
            normalized = np.hstack((labels, content))
            normalized = np.squeeze(np.asarray(normalized))
        else:
            pdb.set_trace()
            normalized = preprocessing.scale(data)



        return normalized


    def saveWindows(self, windows, data_dir, curri):
        '''
        Serializes and saves windows using pickle
        @:param windows: Object with windows to be saved to disk
        '''

        for i, window in enumerate(tqdm.tqdm(windows)):
            label = self.getMostCommonClass(window)
            data = window[:, 1:]
            obj = {"data": data, "label": label}
            f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri + i)), 'wb')
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        print('[WindowGen] - Saved windows {}'.format(curri + i))

        return curri + i

    def saveMarkerWindows(self, d_wins, l_wins, data_dir, curri):
        '''
        Serializes and saves windows using pickle
        @:param windows: Object with windows to be saved to disk
        '''

        for i, (window_data, label_data) in enumerate(tqdm.tqdm(zip(d_wins, l_wins), total=d_wins.shape[0])):
            label = self.getMostCommonClass(label_data)
            obj = {"data": window_data, "label": label}
            f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri + i)), 'wb')
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        print('[WindowGen] - Saved windows {}'.format(curri + i))

        return curri + i

    def checkDirExists(self, dir):
        dir = dir.format(self.win_size,
                         self.win_stride,
                         "")
        return os.path.isdir(dir)


    def runMarkers(self):
        """
        Main function to generate new windows for skeleton dataset
        :return: Dataset with windows exists
        """

        labels = ['train', 'validate', 'test']

        if self.checkDirExists(self.save_marker_dataset_dir):
            print('[WindowGen] - Dataset already exists, skipping generation...')
            return True
        else:
            for folder in labels:
                os.makedirs(self.save_marker_dataset_dir.format(self.win_size,
                                                                self.win_stride,
                                                                folder))
        markers_dict = dict(
            train=['02', '06'],
            validate=['04'],
            test=['03']
        )

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(markers_dict[folder]):
                skeletondir = 'P{}'.format(dir)
                files = glob.glob(self.dataset_dir.format(skeletondir))
                for k, file in enumerate(files):
                    print('[WindowGen] - Saving for found file {}'.format(file))
                    # pdb.set_trace()
                    try:
                        markerseq = re.search('P[0-9]*_R(.+?)_A[0-9]*', file).group(1)
                        markerfile = self.markerset_dir.format(dir, 13)
                        skdata = self.read_data(file)
                        mkdata = self.read_data_markers(markerfile).astype('float64')
                        labels = skdata[:,0].reshape((-1,1))
                        nanfilter = np.isnan(mkdata).any(axis=1)
                        labels = labels[~nanfilter]
                        mkdata = mkdata[~nanfilter]
                        mergedata = np.hstack((labels, mkdata))     # Combined markers with labels
                        filteredData = self.removeClass(mkdata, 7)   # Removed unused 7 class
                        normalizedData = self.normalizeData(filteredData, haslabels=False)   # Normalize data per sensor channel
                        stackedData = self.coords_to_channels(normalizedData)   # (X,Y,Z) coords to Channels(dims)

                        data_windows = sliding_window(stackedData,
                                                      (stackedData.shape[0], self.win_size, stackedData.shape[2]),
                                                      (1, self.win_stride, 1))
                        label_windows = sliding_window(labels,
                                                       (self.win_size, labels.shape[1]),
                                                       (self.win_stride, 1))

                        win_amount = self.saveMarkerWindows(data_windows,
                                                            label_windows,
                                                            self.save_marker_dataset_dir.format(self.win_size,
                                                                                         self.win_stride,
                                                                                         folder),
                                                            win_amount)
                    except AttributeError:
                        print('something wrong on regexp side')

        end = time.time()
        t = end - start
        print('[WindowGen] - Process has been finished after: {}'.format(t))

        return True
                    


    def run(self):
        """
        Main function to generate new windows for skeleton dataset
        :return: Dataset with windows exists
        """

        labels = ['train', 'validate', 'test']

        if self.checkDirExists(self.save_dataset_dir):
            print('[WindowGen] - Dataset already exists, skipping generation...')
            return True
        else:
            for folder in labels:
                os.makedirs(self.save_dataset_dir.format(self.win_size,
                                                         self.win_stride,
                                                         folder))

        dataset_dict = dict(
            train=['P02', 'P06'],
            validate=['P04'],
            test=['P03']
        )

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(dataset_dict[folder]):
                print('[WindowGen] - Saving for person {}'.format(dir))
                files = glob.glob(self.dataset_dir.format(dir))
                for k, file in enumerate(files):
                    print('[WindowGen] - Saving for found file {}'.format(file))
                    rawData = self.read_data(file)
                    filteredData = self.removeClass(rawData, 7)
                    normalizedData = self.normalizeData(filteredData)
                    data_windows = sliding_window(normalizedData,
                                                  (self.win_size, normalizedData.shape[1]),
                                                  (self.win_stride, 1))
                    win_amount = self.saveWindows(data_windows,
                                             self.save_dataset_dir.format(self.win_size,
                                                                      self.win_stride,
                                                                      folder),
                                             win_amount)

        end = time.time()
        t = end - start
        print('[WindowGen] - Process has been finished after: {}'.format(t))

        return True

if __name__ == '__main__':
    window_size = 100
    window_step = 1

    wg = WindowGenerator(window_size, window_step)
    wg.runMarkers()
    # wg.run()



# EXTRA CODE USED TO PLOT SKELETON

# def extractPose(sequence, i):
#     '''
#     Extract pose on a specific time instance i
#     @:param sequence: sequence of positions
#     @:param i: time instance i
#     '''
#
#     t = i*2
#     n = 134
#     poseData = sequence.iloc[t:(t+1), 2:n].to_numpy()
#     pose = poseData.astype('float').reshape(-1,3)[1::2]
#     print(pose)
#
#     return pose


# def printPose(rawData, i):
#     '''
#     Print extracted pose from specific instance i
#     @:param Pose positions
#     '''
#     pose  = extractPose(rawData, i)
#
#     line_pairs = [(1, 0), (0, 2), (2, 6), (2, 0), (2, 16), (2, 11), (11, 21), (16, 13), (13, 19), (19, 20), (6, 3),
#                   (3, 9), (9, 10), (21, 4), (21, 14), (14, 17), (17, 15), (15, 18), (4, 7), (7, 5), (5, 8)]
#
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pose[:, 0],pose[:, 1],pose[:, 2], marker='o', c='r')
#
#     for a,b in line_pairs:
#         x = pose[[a, b], 0]
#         y = pose[[a, b], 1]
#         z = pose[[a, b], 2]
#         ax.plot(x, y, z, 'go-', linewidth='2')
#
#     plt.show()

