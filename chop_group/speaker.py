from chop_group.random_cycler import RandomCycler
from chop_group.utterance import Utterance
import os
import pickle
# from pathlib import Path

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, speaker_dict_file, proc_data_path):
        self.speaker_id = speaker_dict_file.split(".")[0]
        self.speaker_dict_path = os.path.join(proc_data_path, speaker_dict_file)
        self.utterances = None
        self.utterance_cycler = None
        self.speaker_dict = None
        
    def _load_utterances(self):
        self.speaker_dict = pickle.load(open(self.speaker_dict_path, "rb"))
        self.utterances = [Utterance(np_frames) for np_frames in list(self.speaker_dict.values())]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
