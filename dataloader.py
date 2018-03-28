import os
from pickle import load

class Dataloader(object):
    """Loader object for datasets"""
    def __init__(self, dataset_url):
        self.path = os.path.abspath(__file__)
        self.folder = self.path.replace("dataloader.pyc", "").replace("dataloader.py", "")
        self.dataset_folder = dataset_url.split('/')[-1]
        self.location = os.path.join(self.folder, 'data', self.dataset_folder, 'pickled')
        self.dataset = dataset_url.lstrip('/')
    
    def get_feature(self, feature):
        """The unified API for getting specified features"""
        feature_path = os.path.join(self.location, feature + '.pkl')
        print(feature_path)
        feature_present = os.path.exists(feature_path)
        if not feature_present:
            downloaded = download(self.dataset, feature, self.location)
            if not downloaded:
                return None

        # TODO: check MD5 values and etc. to ensure the downloaded dataset's intact
        with open(feature_path, 'rb') as fp:
            try:
                u = pickle._Unpickler(fp)
                u.encoding = 'latin1'
                feature_values = u.load()
            except:
                print("The previously downloaded dataset is compromised, downloading a new copy...")
                dowloaded = download(self.dataset, feature, self.location)
                if not downloaded:
                    return None
        return feature_values

    def facet(self):
        """Returns a single-field dataset object for facet features"""
        facet_values = self.get_feature('facet')
        return facet_values

    def openface(self):
        """Returns a single-field dataset object for openface features"""
        openface_values = self.get_feature('openface')
        return openface_values

    def embeddings(self):
        """Returns a single-field dataset object for embeddings"""
        embeddings_values = self.get_feature('embeddings')
        return embeddings_values

    def words(self):
        """Returns a single-field dataset object for one-hot vectors of words"""
        words_values = self.get_feature('words')
        return words_values

    def phonemes(self):
        """Returns a single-field dataset object for one-hot vectors of phonemes"""
        phonemes_values = self.get_feature('phonemes')
        return phonemes_values

    def covarep(self):
        """Returns a single-field dataset object for covarep features"""
        covarep_values = self.get_feature('covarep')
        return covarep_values

    def opensmile(self):
        """Returns a single-field dataset object for opensmile features"""
        opensmile_values = self.get_feature('opensmile')
        return opensmile_values

    def sentiments(self):
        """Returns a nested dictionary that stores the sentiment values"""
        sentiments_values = self.get_feature('sentiments')
        return sentiments_values

    def emotions(self):
        """Returns a nested dictionary that stores the emotion distributions"""
        emotions_values = self.get_feature('emotions')
        return emotions_values

    def train(self):
        """Returns three sets of video ids: train, dev, test"""
        train_ids = self.get_feature('train')
        return train_ids

    def valid(self):
        """Returns three sets of video ids: train, dev, test"""
        valid_ids = self.get_feature('valid')
        return valid_ids

    def test(self):
        """Returns three sets of video ids: train, dev, test"""
        test_ids = self.get_feature('test')
        return test_ids

    def original(self, dest):
        """Downloads the dataset files as a tar ball, to the specified destination"""
        # raw_path = os.path.join(dest, self.dataset + '.tar')
        downloaded = download_raw(self.dataset, dest)


class MOSI(Dataloader):
    """Dataloader for CMU-MOSI dataset"""
    def __init__(self):
        super(MOSI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
        print("This API will be deprecated in the future versions. Please check the Github page for the current API")


class MOSEI(Dataloader):
    """Dataloader for CMU-MOSEI dataset"""
    def __init__(self):
        super(MOSEI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')
        print("This API will be deprecated in the future versions. Please check the Github page for the current API")
        
import numpy as np
import urllib
import sys
import os
from subprocess import call

p2fa_phonemes = [ "EH2", "K", "S", "L", "AH0", "M", "EY1", "SH", "N", "P", "OY2", "T", "OW1", "Z", "W", "D", "AH1", "B", "EH1", "V", "IH1", "AA1", "R", "AY1", "ER0", "AE1", "AE2", "AO1", "NG", "G", "IH0", "TH", "IY2", "F", "DH", "IY1", "HH", "UH1", "IY0", "OY1", "OW2", "CH", "UW1", "IH2", "EH0", "AO2", "AA0", "AA2", "OW0", "EY0", "AE0", "AW2", "AW1", "EY2", "UW0", "AH2", "UW2", "AO0", "JH", "Y", "ZH", "AY2", "ER1", "UH2", "AY0", "ER2", "OY0", "UH0", "AW0", "br", "cg", "lg", "ls", "ns", "sil", "sp" ]

def phoneme_index(phoneme):
    return p2fa_phonemes.index(phoneme)

def phoneme_hotkey_enc(phoneme):
    index = phoneme_index(phoneme)
    enc = np.zeros(len(p2fa_phonemes))
    enc[index] = 1
    return enc

def download(dataset, feature, dest):
    call(['mkdir', '-p', dest])
    url = dataset + '/' + feature + '.pkl'
    file_path = os.path.join(dest, feature + '.pkl')
    print(file_path)

    try:
        u = urllib.request.urlopen(url)
    except urllib.request.HTTPError:
        print("The requested data is not available for {} dataset.".format(dataset))
        return False
    with open(file_path, 'wb') as f:
        meta = u.info()
        file_size = int(u.getheader("Content-Length"))
        print("Downloading: {}, size: {}".format(' '.join([dataset, feature]), file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                print('wtf')
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] [%3.2f%%]" % ('='*int((file_size_dl * 100. / file_size)/5), file_size_dl * 100. / file_size))
            sys.stdout.flush()
    sys.stdout.write('\n')
    return True


import pickle
import numpy as np
from scipy.io import loadmat
import pandas as pd
import warnings
from collections import OrderedDict
from copy import deepcopy


class Dataset(object):
    """Primary class for loading and aligning dataset"""

    def __init__(self, dataset_file='', stored=False):
        """
        Initialise the Dataset class. Support two loading mechanism - 
        from dataset files and from the pickle file, decided by the param
        stored.
        :param stored: True if loading from pickle, false if loading from 
                       dataset feature files. Default False
        :param dataset_file: Filepath to the file required to load dataset 
                             features. CSV or pickle file depending upon the
                             loading mechanism
        :timestamps: absolute or relative.
        """
        self.feature_dict = None
        self.timestamps = 'absolute' # this is fixed, we no longer support relative timestamps
        self.stored = stored
        self.dataset_file = dataset_file
        self.phoneme_dict = p2fa_phonemes
        self.loaded = False

    def __getitem__(self, key):
        """Adding direct access of internal data"""
        return self.feature_dict[key]

    def keys(self):
        """Wrapper for .keys() for the feature_dict"""
        return self.feature_dict.keys()

    def items(self):
        """Wrapper for .items() for the feature_dict"""
        return self.feature_dict.items()

    def load(self):
        """
        Loads feature dictionary for the input dataset
        :returns: Dictionary of features for the dataset with each modality 
         as dictionary key
        """

        # Load from the pickle file if stored is True
        if self.stored:
            self.dataset_pickle = self.dataset_file
            with open(self.dataset_pickle) as fp:
                u = pickle._Unpickler(p)
                u.encoding = 'latin1'
                self.feature_dict = u.load()
            return self.feature_dict

        # Load the feature dictionary from the dataset files
        self.dataset_csv = self.dataset_file
        self.feature_dict = self.controller()
        self.loaded = True
        return self.feature_dict

    def controller(self):
        """
        Validates the dataset csv file and loads the features for the dataset
        from its feature files
        """

        def validate_file(self):
            data = pd.read_csv(self.dataset_csv, header=None)
            data = np.asarray(data)
            #data = data[:,:7]
            self.dataset_info = {}
            modality_count = len(data[0]) - 4
            self.modalities = {}
            for i in range(modality_count):
                # key = 'modality_' + str(i)
                key = str(data[0][i + 4])
                info = {}
                info["level"] = str(data[1][i + 4])
                info["type"] = str(data[0][i + 4])
                self.modalities[key] = info

            for record in data[2:]:
                video_id = str(record[0])
                segment_id = str(record[1])
                if video_id not in self.dataset_info:
                    self.dataset_info[video_id] = {}
                if segment_id in self.dataset_info[video_id]:
                    raise NameError("Multiple instances of segment "
                                    + segment_id + " for video " + video_id)
                segment_data = {}
                segment_data["start"] = float(record[2])
                segment_data["end"] = float(record[3])
                for i in range(modality_count):
                    # key = 'modality_' + str(i)
                    key = str(data[0][i + 4])
                    segment_data[key] = str(record[i + 4])
                self.dataset_info[video_id][segment_id] = segment_data
            return

        def load_features(self):
            feat_dict = {}
            data = self.dataset_info
            modalities = self.modalities
            timestamps = self.timestamps
            for key, value in modalities.iteritems():
                api = value['type']
                level = value['level']
                loader_method = Dataset.__dict__["load_" + api]
                modality_feats = {}
                print("Loading features for", api)
                for video_id, video_data in data.iteritems():
                    video_feats = {}
                    for segment_id, segment_data in video_data.iteritems():
                        filepath = str(segment_data[key])
                        start = segment_data["start"]
                        end = segment_data["end"]
                        video_feats[segment_id] = loader_method(self,
                                                                filepath, start,
                                                                end, timestamps=self.timestamps,
                                                                level=level)
                    modality_feats[video_id] = video_feats
                    modality_feats = OrderedDict(sorted(modality_feats.items(), key=lambda x: x[0]))
                feat_dict[key] = modality_feats

            return feat_dict

        validate_file(self)
        feat_dict = load_features(self)

        return feat_dict

    def load_opensmile(self, filepath, start, end, timestamps='absolute', level='s'):
        """
        Load OpenSmile Features from the file corresponding to the param
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        Note: Opensmile support features for entire segment or video only and 
              will return None if level is 'v' and start time is 
        """
        features = []
        start_time, end_time = start, end
        if timestamps == 'relative':
            start_time = 0.0
            end_time = end - start

        if level == 's' or start == 0.0:
            feats = open(filepath).readlines()[-1].strip().split(',')[1:]
            feats = [float(feat_val) for feat_val in feats]
            feat_val = np.asarray(feats, dtype=np.float32)
            features.append((start_time, end_time, feat_val))
        else:
            print("Opensmile support features for the entire segment")
            return None
        return features

    def load_covarep(self, filepath, start, end, timestamps='absolute', level='s'):
        """
        Load COVAREP Features from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []
        time_period = 0.01
        f_content = loadmat(filepath)
        feats = f_content['features']
        start_time, end_time = start, end
        if timestamps == "relative":
            start_time, end_time = 0.0, end - start

        if level == 's':
            feat_start = start_time
            for feat in feats:
                feat_end = feat_start + time_period
                feat_val = np.asarray(feat)
                features.append((max(feat_start - start_time, 0), max(feat_end - start_time, 0), feat_val))
                feat_start += time_period
        else:
            feat_count = feats.shape[0]
            start_index = int(min((start / time_period), feat_count))
            end_index = int(min((end / time_period), feat_count))
            feat_start = start_time
            for feat in feats[start_index:end_index]:
                feat_end = feat_start + time_period
                feat_val = np.asarray(feat)
                features.append((max(feat_start - start_time, 0), max(feat_end - start_time, 0), feat_val))
                feat_start += time_period
        return features

    def load_phonemes(self, filepath, start, end, timestamps='relative', level='s'):
        """
        Load P2FA phonemes as Features from the file corresponding to the 
        param filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []
        start_time, end_time = start, end
        if timestamps == "relative":
            start_time, end_time = 0.0, end - start

        if level == 's':
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_val = [float(val) for val in line.split(",")[2:]]
                    feat_val = np.asarray(feat_val)
                    features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        else:
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_time = feat_end - feat_start
                    if ((feat_start <= start and feat_end > end)
                        or (feat_start >= start and feat_end < end)
                        or (feat_start <= start
                            and start - feat_start < feat_time / 2)
                        or (feat_start >= start
                            and end - feat_start > feat_time / 2)):

                        feat_start = feat_start - start
                        feat_end = feat_end - start
                        feat_val = [float(val) for val in line.split(",")[2:]]
                        feat_val = np.asarray(feat_val)
                        features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        return features

    def load_embeddings(self, filepath, start, end, timestamps='relative', level='s'):
        """
        Load Word Embeddings from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []
        start_time, end_time = start, end
        if timestamps == "relative":
            start_time, end_time = 0.0, end - start

        if level == 's':
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_val = [float(val) for val in line.split(",")[2:]]
                    feat_val = np.asarray(feat_val)
                    features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        else:
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_time = feat_end - feat_start
                    if ((feat_start <= start and feat_end > end)
                        or (feat_start >= start and feat_end < end)
                        or (feat_start <= start
                            and start - feat_start < feat_time / 2)
                        or (feat_start >= start
                            and end - feat_start > feat_time / 2)):

                        feat_start = feat_start - start
                        feat_end = feat_end - start
                        feat_val = [float(val) for val in line.split(",")[2:]]
                        feat_val = np.asarray(feat_val)
                        features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        return features

    def load_words(self, filepath, start, end, timestamps='relative', level='s'):
        """
        Load one hot embeddings for words as features from the file 
        corresponding to the param filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []
        start_time, end_time = start, end
        if timestamps == "relative":
            start_time, end_time = 0.0, end - start

        if level == 's':
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_val = [float(val) for val in line.split(",")[2:]]
                    feat_val = np.asarray(feat_val)
                    #print (feat_start, feat_end)
                    #assert False
                    features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        else:
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines():
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[0])
                    feat_end = float(line.split(",")[1])
                    feat_time = feat_end - feat_start
                    if ((feat_start <= start and feat_end > end)
                        or (feat_start >= start and feat_end < end)
                        or (feat_start <= start
                            and start - feat_start < feat_time / 2)
                        or (feat_start >= start
                            and end - feat_start > feat_time / 2)):

                        feat_start = feat_start - start
                        feat_end = feat_end - start
                        feat_val = [float(val) for val in line.split(",")[2:]]
                        feat_val = np.asarray(feat_val)
                        features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        return features

    def load_openface(self, filepath, start, end, timestamps='absolute', level='s'):
        """
        Load OpenFace features from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []
        time_period = 0.0333333

        start_time, end_time = start, end
        if timestamps == "relative":
            start_time, end_time = 0.0, end - start

        if level == 's':
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines()[1:]:
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[1])
                    feat_end = feat_start + time_period
                    feat_val = [float(val) for val in line.split(",")[2:]]
                    feat_val = np.asarray(feat_val, dtype=np.float32)
                    features.append((max(feat_start, 0), max(feat_end, 0), feat_val))

        else:
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines()[1:]:
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[1])

                    if (feat_start >= start and feat_start < end):
                        # To adjust the timestamps
                        feat_start = feat_start - start
                        feat_end = feat_start + time_period
                        feat_val = [float(val) for val in line.split(",")[2:]]
                        feat_val = np.asarray(feat_val, dtype=np.float32)
                        features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        return features
    
    # note that this is implicity new facet
    def load_facet(self, filepath, start, end, timestamps='absolute', level='v'):
        """
        Load FACET features from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        features = []

        # load a subset of current segment and infer its format
        start_row = 0
        start_col = 0
        with open(filepath, 'r') as f_handle:
            splitted = []
            for line in f_handle.readlines()[0:10]:
                splitted.append(line.split(","))

            # check if the first row is a header by checking if the first field is a number
            try:
                float(splitted[start_row][start_col])
            except:
                start_row = 1

            # check if the first column is a index column by checking if it increments by 1 everytime
            for i in range(1, len(splitted) - 1):
                if (float(splitted[i+1][0]) - float(splitted[i][0])) != 1:
                    start_col = 0
                    break
                start_col = 1

            time_period = float(splitted[start_row][start_col])

        start_time, end_time = start, end
        # if timestamps == "relative":
        #     start_time, end_time = 0.0, end - start

        if level == 's':
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines()[start_row:]:
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[start_col])
                    feat_end = feat_start + time_period
                    feat_val = []
                    for val in line.split(",")[start_col + 1:-1]:
                        try:
                            feat_val.append(float(val))
                        except:
                            feat_val.append(0.0)
                    feat_val = np.asarray(feat_val, dtype=np.float32)
                    features.append((max(feat_start, 0), max(feat_end, 0), feat_val))

        else:
            with open(filepath, 'r') as f_handle:
                for line in f_handle.readlines()[start_row:]:
                    line = line.strip()
                    if not line:
                        break
                    feat_start = float(line.split(",")[start_col])

                    if (feat_start >= start and feat_start < end):
                        # To adjust the timestamps
                        feat_start = feat_start - start
                        feat_end = feat_start + time_period
                        feat_val = []
                        for val in line.split(",")[start_col + 1:-1]:
                            try:
                                feat_val.append(float(val))
                            except:
                                feat_val.append(0.0)
                        feat_val = np.asarray(feat_val, dtype=np.float32)
                        features.append((max(feat_start, 0), max(feat_end, 0), feat_val))
        return features

    def load_facet1(self, filepath, start, end, timestamps='absolute', level='v'):
        """
        Load FACET features from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        return self.load_facet(filepath, start, end, timestamps=timestamps, level=level)


    def load_facet2(self, filepath, start, end, timestamps='absolute', level='v'):
        """
        Load FACET features from the file corresponding to the param 
        filepath
        :param start: Start time of the segment
        :param end: End time of the segment
        :param filepath: Path to the opensmile feature files
        :param level: 's' if the file contains features only for the segment,
                      i.e. interval (start, end), 'v' if for the entire video 
        :param timestamps: relative or absolute
        :returns: List of tuples (feat_start, feat_end, feat_value)
                  corresponding to the features in the interval.
        """
        return self.load_facet(filepath, start, end, timestamps=timestamps, level=level)

    def align(self, align_modality):
        aligned_feat_dict = {}
        modalities = self.modalities
        alignments = self.get_alignments(align_modality)
        for modality in modalities:
            # if modality == align_modality:
            #     continue
            aligned_modality = self.align_modality(modality, alignments)
            aligned_feat_dict[modality] = OrderedDict(sorted(aligned_modality.items(), key=lambda x: x[0]))
        self.aligned_feature_dict = aligned_feat_dict
        return aligned_feat_dict

    def get_alignments(self, modality):
        alignments = {}
        aligned_feat_dict = self.feature_dict[modality]

        for video_id, segments in aligned_feat_dict.iteritems():
            segment_alignments = {}
            for segment_id, features in segments.iteritems():
                segment_alignments[segment_id] = []
                for value in features:
                    timing = (value[0], value[1])
                    segment_alignments[segment_id].append(timing)
            alignments[video_id] = segment_alignments
        return alignments

    def align_modality(self, modality, alignments, merge_type="mean"):
        aligned_feat_dict = {}
        modality_feat_dict = self.feature_dict[modality]
        warning_hist = set() # Keep track of all the warnings

        for video_id, segments in alignments.iteritems():
            aligned_video_feats = {}

            for segment_id, feat_intervals in segments.iteritems():
                aligned_segment_feat = []

                for start_interval, end_interval in feat_intervals:
                    time_interval = end_interval - start_interval
                    feats = modality_feat_dict[video_id][segment_id]
                    try:
                        aligned_feat = np.zeros(len(feats[0][2]))
                    except:
                        if (video_id, segment_id) not in warning_hist:
                            print("\nModality {} for video {} segment {} is (partially) missing and is thus being replaced by zeros!\n".format(modality.split("_")[-1], video_id, segment_id))
                            warning_hist.add((video_id, segment_id))
                        # print modality, video_id, segment_id, feats
                        for sid, seg_data in modality_feat_dict[video_id].items():
                            if seg_data != []:
                                feats = seg_data
                                break
                    try: 
                        aligned_feat = np.zeros(len(feats[0][2]))
                    except:
                        aligned_feat = np.zeros(0)

                    for feat_tuple in feats:
                        feat_start = feat_tuple[0]
                        feat_end = feat_tuple[1]
                        feat_val = feat_tuple[2]
                        if (feat_start < end_interval
                                and feat_end >= start_interval):
                            feat_weight = (min(end_interval, feat_end) -
                                           max(start_interval, feat_start)) / time_interval
                            weighted_feat = np.multiply(feat_val, feat_weight)
                            if np.shape(aligned_feat) == (0,):
                                aligned_feat = weighted_feat
                            else:
                                aligned_feat = np.add(aligned_feat, weighted_feat)

                    aligned_feat_tuple = (start_interval, end_interval,
                                          aligned_feat)
                    aligned_segment_feat.append(aligned_feat_tuple)
                aligned_video_feats[segment_id] = aligned_segment_feat
            aligned_feat_dict[video_id] = aligned_video_feats

        return aligned_feat_dict

    @staticmethod
    def merge(dataset1, dataset2):
        # ensure the merged objects are indeed Datasets
        assert isinstance(dataset1, Dataset)
        assert isinstance(dataset2, Dataset)
        # merge the feature_dict and modalities attributes
        merged_modalities = Dataset.merge_dict(dataset1.modalities, dataset2.modalities)
        merged_feat_dict = Dataset.merge_dict(dataset1.feature_dict, dataset2.feature_dict)
        mergedDataset = Dataset()
        mergedDataset.feature_dict = merged_feat_dict
        mergedDataset.modalities = merged_modalities
        return mergedDataset


    @staticmethod
    def merge_dict(dict1, dict2):
        merged = deepcopy(dict1)
        merged.update(dict2)
        return merged
