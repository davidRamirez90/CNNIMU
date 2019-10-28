from __future__ import print_function
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import os
import pandas as pd
import logging
import pickle
from sliding_window import sliding_window
from datasetAnalyzer import StatAnalyzer
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

    def __init__(self, win_size=100, win_stride=1, channels=38):
        self.win_size = win_size
        self.win_stride = win_stride

        self.dataset_dir = env.dataset_url
        self.save_dataset_dir = env.window_url

        self.markerset_dir = env.markers_url
        self.save_marker_dataset_dir = env.marker_window_url

        self.save_accel_dataset_dir = env.accel_window_url

        self.imuset_dir = env.imu_url
        self.save_imu_dataset_dir = env.imu_window_url

        self.channels = channels
        self.probabilities = [0, 0.5, 0.83, 0, 1, 0, 0]
        self.new_mk_url = env.new_mk_url
        self.new_sk_url = env.new_sk_url


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


    def read_data_for_imu(self, path):
        '''
        Reads CSV file from path
        @:param File path
        '''
        data = pd.read_csv(path)
        data = data.dropna(axis=1, how='all')
        data = data.iloc[:, 1:]
        data = data.iloc[::2, :]
        data = data.to_numpy()
        return data


    def read_data_markers(self, path):
        '''
        Reads CSV file from path
        @:param File path
        '''
        data = pd.read_csv(path, skiprows=2)
        # pdb.set_trace()
        data = data.dropna(axis=1, how='all')
        data = data.iloc[2:-2, 2:]
        data = pd.concat([data.iloc[:, :57], data.iloc[:, 60:]], axis=1)
        data = data.to_numpy()

        return data

    def read_imu_data(self, path):
        # pdb.set_trace()
        data = pd.read_csv(path, skiprows=2)
        data = data.dropna(axis=1, how='all')
        data = data.iloc[1:, 1:]
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


    def removeClassMarkers(self, data, labels, classN):
        '''
        :param data: Data to remove class from
        :param classN: Class number to remove from data
        :return: filtArray: Filtered Array
        '''
        indices = labels[:, 0] != classN
        filtData = data[indices, :]
        filtLabels = labels[indices, :]

        return filtData, filtLabels


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
            # pdb.set_trace()
            # base = data[:, 6:9]   # Double or legs
            base = data[:, 15:18] # Full data
            conv = np.tile(base, self.channels)
            relative = data-conv
            normalized = preprocessing.scale(relative)



        return normalized


    def saveWindows(self, windows, data_dir, curri, folder):
        '''
        Serializes and saves windows using pickle
        @:param windows: Object with windows to be saved to disk
        '''
        
        for i, window in enumerate(tqdm.tqdm(windows)):
            label = self.getMostCommonClass(window)
            data = window[:, 1:]
            obj = {"data": data, "label": label}
            f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri)), 'wb')
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            curri += 1

            # SAMPLING FRON RANDOM DIST AND DECIDING IF WINDOW SHOULD BE RESAMPLED
            # if folder == "train":
            #     randSample = rand.random_sample()
            #     if (randSample > self.probabilities[int(label)]):
            #         data += rand.normal(0, 1e-2, data.shape)
            #         obj = {"data": data, "label": label}
            #         f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri)), 'wb')
            #         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            #         f.close()
            #         curri += 1

        print('[WindowGen] - Saved windows {}'.format(curri))

        return curri

    def saveMarkerWindows(self, d_wins, l_wins, data_dir, curri, folder):
        '''
        Serializes and saves windows using pickle
        @:param windows: Object with windows to be saved to disk
        '''

        for i, (window_data, label_data) in enumerate(tqdm.tqdm(zip(d_wins, l_wins), total=d_wins.shape[0])):

            label = self.getMostCommonClass(label_data)
            obj = {"data": window_data, "label": label}
            f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri)), 'wb')
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            curri += 1

            # SAMPLING FRON RANDOM DIST AND DECIDING IF WINDOW SHOULD BE RESAMPLED
            # if folder == "train":
            #     randSample = rand.random_sample()
            #     if(randSample > self.probabilities[int(label)]):
            #         window_data += rand.normal(0, 1e-2, window_data.shape)
            #         obj = {"data": window_data, "label": label}
            #         f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(curri)), 'wb')
            #         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            #         f.close()
            #         curri += 1




        print('[WindowGen] - Saved windows {}'.format(curri))

        return curri

    def checkDirExists(self, dir):
        dir = dir.format(self.win_size,
                         self.win_stride,
                         "")
        return os.path.isdir(dir)


    def runMarkers(self):
        """
        Main function to generate new windows for marker dataset
        :return: Dataset with windows exists
        """

        # labels = ['train', 'validate', 'test']
        labels = ['train', 'test']
        # labels = ['train']
        # labels = ['test']

        for folder in labels:
            if self.checkDirExists(self.save_marker_dataset_dir + folder):
                print('[WindowGen] - {} folder already exists, skipping generation...'.format(folder))
                return True
            else:
                os.makedirs(self.save_marker_dataset_dir.format(self.win_size,
                                                         self.win_stride,
                                                         folder))
        # markers_dict = dict(
        #     train=['01', '02', '03'],
        #     validate=['04'],
        #     test=['05', '06']
        # )

        # markers_dict = dict(
        #     train=['01', '02', '03', '04'],
        #     test=['05', '06']
        # )

        markers_dict = dict(
            train=['01', '02', '03', '04'],
            test=['07', '08', '09', '10', '11', '12', '13']
        )


        seenSequences = {
            "01": list(),
            "02": list(),
            "03": list(),
            "04": list(),
            "07": list(),
            "08": list(),
            "09": list(),
            "10": list(),
            "11": list(),
            "12": list(),
            "13": list()
        }

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(markers_dict[folder]):
                skeletondir = 'P{}'.format(dir)
                if folder == "test":
                    files = glob.glob(self.new_sk_url.format(skeletondir))
                else:
                    files = glob.glob(self.dataset_dir.format(skeletondir))

                for k, file in enumerate(files):
                    # pdb.set_trace()
                    try:
                        markerseq = re.search('P[0-9]*_R(.+?)_A[0-9]*', file).group(1)

                        if markerseq not in seenSequences[dir]:
                            print('[WindowGen] - Saving for found file {}'.format(file))
                            # FIRST TIME WE SEE THIS RECORDING FOR THIS SPECIFIC PERSON E.G. R01, R02
                            seenSequences[dir].append(markerseq)
                            if folder == 'test':
                                markerfile = self.new_mk_url.format(dir, markerseq)
                            else:
                                markerfile = self.markerset_dir.format(dir, markerseq)

                            skdata = self.read_data(file)
                            mkdata = self.read_data_markers(markerfile).astype('float64')
                            mkdata = mkdata[:skdata.shape[0], :]
                            labels = skdata[:mkdata.shape[0], 0].reshape((-1, 1))
                            nanfilter = np.isnan(mkdata).any(axis=1)
                            labels = labels[~nanfilter]
                            mkdata = mkdata[~nanfilter]
                            mergedata = np.hstack((labels, mkdata))  # Combined markers with labels
                            filteredData, filteredLabels = self.removeClassMarkers(mkdata, labels,
                                                                                   7)  # Removed unused 7 class
                            if filteredData.shape[0] == 0:
                                break;
                            normalizedData = self.normalizeData(filteredData,
                                                                haslabels=False)  # Normalize data per sensor channel
                            stackedData = self.coords_to_channels(normalizedData)  # (X,Y,Z) coords to Channels(dims)
                            # pdb.set_trace()
                            data_windows = sliding_window(stackedData,
                                                          (stackedData.shape[0], self.win_size, stackedData.shape[2]),
                                                          (1, self.win_stride, 1))
                            label_windows = sliding_window(filteredLabels,
                                                           (self.win_size, labels.shape[1]),
                                                           (self.win_stride, 1))

                            win_amount = self.saveMarkerWindows(data_windows,
                                                                label_windows,
                                                                self.save_marker_dataset_dir.format(self.win_size,
                                                                                                    self.win_stride,
                                                                                                    folder),
                                                                win_amount,
                                                                folder)
                        # else:
                            # print("Skipping person {} for seq {}, cause ive already seen it".format(dir, markerseq))

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

        # labels = [ 'train', 'validate', 'test']
        labels = ['train', 'test']
        # labels = ['train']
        # labels = ['test']


        for folder in labels:
            if self.checkDirExists(self.save_dataset_dir+folder):
                print('[WindowGen] - {} folder already exists, skipping generation...'.format(folder))
                return True
            else:
                os.makedirs(self.save_dataset_dir.format(self.win_size,
                                                         self.win_stride,
                                                         folder))


        # if self.checkDirExists(self.save_dataset_dir):
        #     print('[WindowGen] - Dataset already exists, skipping generation...')
        #     return True
        # else:
        #     for folder in labels:
        #         os.makedirs(self.save_dataset_dir.format(self.win_size,
        #                                                  self.win_stride,
        #                                                  folder))

        # dataset_dict = dict(
        #     train=['P01', 'P02', 'P03'],
        #     validate=['P04'],
        #     test=['P05', 'P06']
        # )

        # dataset_dict = dict(
        #     train=['P01', 'P02', 'P03', 'P04'],
        #     test=['P05', 'P06']
        # )

        dataset_dict = dict(
            train=['P01', 'P02', 'P03', 'P04'],
            test=['P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13']
        )

        seenSequences = {
            "P01": list(),
            "P02": list(),
            "P03": list(),
            "P04": list(),
            "P07": list(),
            "P08": list(),
            "P09": list(),
            "P10": list(),
            "P11": list(),
            "P12": list(),
            "P13": list()
        }

        

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(dataset_dict[folder]):
                print('[WindowGen] - Saving for person {}'.format(dir))
                if folder == "test":
                    files = glob.glob(self.new_sk_url.format(dir))
                else:
                    files = glob.glob(self.dataset_dir.format(dir))

                #  pdb.set_trace()
                for k, file in enumerate(files):
                    try:
                        markerseq = re.search('P[0-9]*_R(.+?)_A[0-9]*', file).group(1)

                        if markerseq not in seenSequences[dir]:
                            print('[WindowGen] - Saving for found file {}'.format(file))
                            # FIRST TIME WE SEE THIS RECORDING FOR THIS SPECIFIC PERSON E.G. R01, R02
                            seenSequences[dir].append(markerseq)
                            rawData = self.read_data(file)
                            filteredData = self.removeClass(rawData, 7)
                            if filteredData.shape[0] == 0:
                                break;
                            normalizedData = self.normalizeData(filteredData)
                            data_windows = sliding_window(normalizedData,
                                                          (self.win_size, normalizedData.shape[1]),
                                                          (self.win_stride, 1))
                            win_amount = self.saveWindows(data_windows,
                                                          self.save_dataset_dir.format(self.win_size,
                                                                                       self.win_stride,
                                                                                       folder),
                                                          win_amount,
                                                          folder)
                        # else:
                        #     print("Skipping person {} for seq {}, cause ive already seen it".format(dir, markerseq))

                    except AttributeError:
                        print('something wrong on regexp side')

        end = time.time()
        t = end - start
        print('[WindowGen] - Process has been finished after: {}'.format(t))

        return True

    def plot(self, data, t):
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

    def basicSel(self, data):
        index = [0,2,7,12,13,20,21,25,27,28,33,34,30,31,32,38]
        for i in index:
            try:
                n = data[:, i:i+3]
                prev = np.concatenate((prev, n), axis=1)  # does a exist in the current namespace
            except NameError:
                prev = data[:, i:3]  # nope

        return prev

    def calculate(self, data):
        freq = 200
        dt = 1 / freq
        seconds = data.shape[0] / freq
        time = np.arange(0, seconds, dt).reshape(-1,1)
        time = np.repeat(time, data.shape[1], axis=1)
        pos = data
        vel = self.diff(pos, dt)
        acc = self.diff(vel, dt)

        # self.plot((np.resize(pos, (300, acc.shape[1]))[:,1].reshape(-1,1),
        #            np.resize(vel, (300, acc.shape[1]))[:,1].reshape(-1,1),
        #            np.resize(acc, (300, acc.shape[1]))[:,1].reshape(-1,1)),
        #           np.resize(time, (300, acc.shape[1]))[:,1].reshape(-1,1),
        #           acc)

        return (vel, acc)


    def diff(self, base, dt):
        dbase = list()
        for i, pos in enumerate(base):
            if i == 0:
                prevpos = pos
            else:
                dif = (pos - prevpos) / dt
                dbase.append(dif)
                prevpos = pos
        return np.asarray(dbase)


    def runDerivation(self):
        """
        Main function to generate new windows for marker derivated accel dataset
        :return: Dataset with windows exists
        :return: Dataset with windows exists
        """

        # labels = ['train', 'validate', 'test']
        labels = ['train', 'test']
        # labels = ['test']

        for folder in labels:
            if self.checkDirExists(self.save_accel_dataset_dir+folder):
                print('[WindowGen] - {} folder already exists, skipping generation...'.format(folder))
                return True
            else:
                os.makedirs(self.save_accel_dataset_dir.format(self.win_size,
                                                         self.win_stride,
                                                         folder))

        # markers_dict = dict(
        #     train=['01', '02', '03'],
        #     validate=['04'],
        #     test=['05', '06']
        # )

        # markers_dict = dict(
        #     train=['01', '02', '03', '04'],
        #     test=['05', '06']
        # )

        markers_dict = dict(
            train=['01', '02', '03', '04'],
            test=['07', '08', '09', '10', '11', '12', '13']
        )

        seenSequences = {
            "01": list(),
            "02": list(),
            "03": list(),
            "04": list(),
            "07": list(),
            "08": list(),
            "09": list(),
            "10": list(),
            "11": list(),
            "12": list(),
            "13": list()
        }

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(markers_dict[folder]):
                skeletondir = 'P{}'.format(dir)
                if folder == "test":
                    files = glob.glob(self.new_sk_url.format(skeletondir))
                else:
                    files = glob.glob(self.dataset_dir.format(skeletondir))

                for k, file in enumerate(files):

                    # pdb.set_trace()
                    try:
                        markerseq = re.search('P[0-9]*_R(.+?)_A[0-9]*', file).group(1)

                        if markerseq not in seenSequences[dir]:
                            print('[WindowGen] - Saving for found file {}'.format(file))
                            # FIRST TIME WE SEE THIS RECORDING FOR THIS SPECIFIC PERSON E.G. R01, R02
                            seenSequences[dir].append(markerseq)

                            if folder == 'test':
                                markerfile = self.new_mk_url.format(dir, markerseq)
                            else:
                                markerfile = self.markerset_dir.format(dir, markerseq)

                            skdata = self.read_data(file)
                            mkdata = self.read_data_markers(markerfile).astype('float64')
                            # mkdata = self.basicSel(mkdata)
                            (_, accdata) = self.calculate(mkdata)
                            labels = skdata[0:accdata.shape[0], 0].reshape((-1, 1))
                            nanfilter = np.isnan(accdata).any(axis=1)
                            labels = labels[~nanfilter]
                            accdata = accdata[~nanfilter]
                            filteredData, filteredLabels = self.removeClassMarkers(accdata, labels,
                                                                                   7)  # Removed unused 7 class
                            if filteredData.shape[0] == 0:
                                break;
                            normalizedData = self.normalizeData(filteredData,
                                                                haslabels=False)  # Normalize data per sensor channel
                            stackedData = self.coords_to_channels(normalizedData)  # (X,Y,Z) coords to Channels(dims)
                            # pdb.set_trace()
                            data_windows = sliding_window(stackedData,
                                                          (stackedData.shape[0], self.win_size, stackedData.shape[2]),
                                                          (1, self.win_stride, 1))
                            label_windows = sliding_window(filteredLabels,
                                                           (self.win_size, labels.shape[1]),
                                                           (self.win_stride, 1))

                            win_amount = self.saveMarkerWindows(data_windows,
                                                                label_windows,
                                                                self.save_accel_dataset_dir.format(self.win_size,
                                                                                                    self.win_stride,
                                                                                                    folder),
                                                                win_amount,
                                                                folder)
                    except AttributeError:
                        print('something wrong on regexp side')

        end = time.time()
        t = end - start
        print('[WindowGen] - Process has been finished after: {}'.format(t))

        return True




    def runIMUData(self):
        """
        Main function to generate new windows for IMU dataset
        :return: Dataset with windows exists
        """
        labels = ['train', 'validate', 'test']

        for folder in labels:
            if self.checkDirExists(self.save_imu_dataset_dir + folder):
                print('[WindowGen] - {} folder already exists, skipping generation...'.format(folder))
                return True
            else:
                os.makedirs(self.save_imu_dataset_dir.format(self.win_size,
                                                                self.win_stride,
                                                                folder))

        imu_dict = dict(
            train=['07', '08', '09', '10'],
            validate=['11', '12'],
            test=['12', '13', '14']
        )

        seenSequences = {
            "07": list(),
            "08": list(),
            "09": list(),
            "10": list(),
            "11": list(),
            "12": list(),
            "13": list(),
            "14": list()
        }

        print('[WindowGen] - Creating Training Windows')

        start = time.time()

        for i, folder in enumerate(labels):
            print('[WindowGen] - Saving on folder {}'.format(folder))
            win_amount = 0
            for j, dir in enumerate(imu_dict[folder]):
                skeletondir = 'P{}'.format(dir)
                files = glob.glob(self.new_sk_url.format(skeletondir))
                print(dir)
                # pdb.set_trace()
                for k, file in enumerate(files):
                    try:
                        imuseq = re.search('P[0-9]*_R(.+?)_A[0-9]*', file).group(1)
                        if imuseq not in seenSequences[dir]:
                            print('[WindowGen] - Saving for found file {}'.format(file))
                            seenSequences[dir].append(imuseq)
                            imufile = self.imuset_dir.format(dir, dir, imuseq)
                            print(imufile)
                            if not os.path.isfile(imufile):
                                print("skipping")
                                continue
                            print("generating windows...")
                            skdata = self.read_data_for_imu(file)
                            imudata = self.read_imu_data(imufile).astype('float64')
                            # pdb.set_trace()
                            imudata = imudata[:skdata.shape[0], :]
                            skdata = skdata[:imudata.shape[0], :]
                            labels = skdata[:imudata.shape[0], 0].reshape((-1, 1))
                            nanfilter = np.isnan(imudata).any(axis=1)
                            labels = labels[~nanfilter]
                            imudata = imudata[~nanfilter]
                            filteredData, filteredLabels = self.removeClassMarkers(imudata, labels, 7)

                            an = StatAnalyzer()
                            an.calculate(filteredLabels)
                            pdb.set_trace()

                            if filteredData.shape[0] == 0:
                                continue
                            stackedData = self.coords_to_channels(filteredData)

                            data_windows = sliding_window( stackedData,
                                                           (stackedData.shape[0],
                                                            self.win_size,
                                                            stackedData.shape[2]),
                                                           (1, self.win_stride, 1))
                            label_windows = sliding_window(filteredLabels,
                                                           (self.win_size, labels.shape[1]),
                                                           (self.win_stride, 1))
                            win_amount = self.saveMarkerWindows(data_windows,
                                                             label_windows,
                                                             self.save_imu_dataset_dir.format(self.win_size,
                                                                                              self.win_stride,
                                                                                              folder),
                                                             win_amount,
                                                             folder)
                    except AttributeError:
                        print('something went wrong with regexp')
        end = time.time()
        t = end-start
        print('[WindowGen] - Process has been finished after: {}'.format(t))

        return True
















if __name__ == '__main__':

    # TYPE 0
    # print("Creating Skeletons")
    # wg1 = WindowGenerator(100, 5, 138)
    # wg1.run()
    print("Creating Markers")
    wg2 = WindowGenerator(50, 2, 38)
    # wg2.runMarkers()
    # print("Creating Accel")
    # wg2.runDerivation()
    print("Creating IMU")
    wg2.runIMUData()
    # wg1 = WindowGenerator(300, 15)
    # wg1.runMarkers()
    # wg1.run()
    # wg2.run()
    # wg3.run()
    # wg4.run()
    # wg5.run()
    # wg6.run()



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

