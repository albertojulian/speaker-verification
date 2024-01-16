# import anvil.server
# anvil.server.connect("IRZZXUQ6TMY7GD3BP3WUZ6QR-IQV6DNSXDFAQDPXA")

from feature_extraction import FeatureExtractor
from GE2ELoss import SpeakerEncoder
import yaml
import pickle
import os
import numpy as np
import tensorflow as tf
import random
import time

class SpeakerVerification:
    def __init__(self, speaker_config_dict):
        super().__init__()
        self.speaker_config_dict = speaker_config_dict

        self.rel_path = speaker_config_dict["rel_path"]

        self.enroll_folder = speaker_config_dict["enroll_folder"]
        self.enroll_path = os.path.join(self.rel_path, self.enroll_folder)
        enrolled_speakers = os.listdir(self.enroll_path)
        self.enrolled_speakers = [s for s in enrolled_speakers if s != ".DS_Store"]

        self.verify_folder = speaker_config_dict["verify_folder"]
        self.verify_path = os.path.join(self.rel_path, self.verify_folder)
        verify_speakers = os.listdir(self.verify_path)
        self.verify_speakers = [s for s in verify_speakers if s != ".DS_Store"]

        self.feature_extractor =  FeatureExtractor(speaker_config_dict)
        self.speaker_encoder = None

    def enroll(self, speaker_id=None, youtube_id=None, utter_files=None, n_utterances=4,
               print_mode=True, return_embeddings=False):
        """
        par: Several utterances of one user.
        For each utterance,
        - feature extraction (mel spectrogram) is performed
        - a batch is made applying a sliding window over each transformed utterance
        - the batch feeds the speaker encoder
        - the embeddings at the output are averaged.
        With all the mean embeddings of that user enrollment, the centroid is computed.
        """
        if speaker_id==None:
            speaker_id = self.random_speaker(n_utterances=n_utterances, mode="enroll")
            if speaker_id==None:
                return None

        if youtube_id==None:
            youtube_id = self.random_youtube(speaker_id, n_utterances=n_utterances, mode="enroll")

        if utter_files==None:
            utter_files = self.random_utterances(speaker_id, youtube_id, n_utterances, mode="enroll")

        # if print_mode==True:
        print(f"Enrolling speaker_id: {speaker_id}, with youtube_id: {youtube_id} and utters: {utter_files}")

        embedding_batch = []
        for utter_file in utter_files:
            embedding_mean = self.get_speaker_embedding(speaker_id, youtube_id, utter_file,
                                                        mode="enroll", print_mode=print_mode)
            if embedding_mean != None:
                embedding_batch.append(embedding_mean)
            else:
                print(f"embedding_mean empty for {utter_file=}")
                quit()

        embedding_batch = np.array(embedding_batch)
        # Compute centroid for the embeddings
        centroid = self.speaker_encoder.get_centroid(embedding_batch)

        if return_embeddings==False:
            return centroid
        else:
            return centroid, embedding_batch

    def verify_speaker(self, centroid, speaker_id, youtube_id, utter_file, mode="verify", print_mode=True):
        embedding_mean = self.get_speaker_embedding(speaker_id, youtube_id, utter_file,
                                                    mode=mode, print_mode=print_mode)
        similarity = 0
        if embedding_mean != None:
            similarity = tf.reduce_sum(embedding_mean * centroid)
        """
        speakers_per_batch = 1
        utterances_per_speaker = 1
        sim_matrix = self.speaker_encoder.similarity(embedding_mean,
                                                     speakers_per_batch=speakers_per_batch,
                                                     utterances_per_speaker=utterances_per_speaker,
                                                     center=centroid)
        # TODO: apply some threshold to similarity matrix?
        return sim_matrix
        """
        return similarity

    def get_speaker_embedding(self, speaker_id, youtube_id, utter_file, mode="enroll", print_mode=True):

        sliding_windows = self.feature_extractor.sliding_windows(speaker_id, youtube_id, utter_file,
                                                                 mode=mode, print_mode=print_mode)

        # Create SpeakerEncoder only once
        if self.speaker_encoder==None:
            checkpoint_path = self.speaker_config_dict["checkpoint_path"]
            if not os.path.exists(checkpoint_path):
                print(f"EXIT: Not found model checkpoint folder: {checkpoint_path}")
                return None

            speaker_config_dict_pkl = self.speaker_config_dict["speaker_config_dict_pkl"]
            speaker_config_dict_path = os.path.join(checkpoint_path, speaker_config_dict_pkl)
            # if os.path.exists(speaker_config_dict_path):
            #    self.speaker_config_dict = pickle.load(open(speaker_config_dict_path, "rb"))

            self.speaker_encoder = SpeakerEncoder(self.speaker_config_dict, train_mode=False)

        embedding_mean = self.speaker_encoder.get_speaker_embedding(sliding_windows)

        return embedding_mean

    def random_speaker(self, n_utterances=4, mode="enroll"):
        sel_speakers = self.filter_speakers(n_utterances=n_utterances, mode=mode)

        idx = random.randrange(len(sel_speakers))
        speaker_id = sel_speakers[idx]
        return speaker_id

    def filter_speakers(self, n_utterances, mode="enroll"):
        if mode=="enroll":
            unfiltered_speakers = self.enrolled_speakers
        else:
            unfiltered_speakers = self.verify_speakers

        sel_speakers = [s for s in unfiltered_speakers if
                        self.filter_youtube(s, n_utterances=n_utterances, mode=mode) != []]
        if len(sel_speakers)==0:
            print(f"ERROR: no enrolled speakers have {n_utterances} utterances or more")
            return None

        return sel_speakers

    def random_youtube(self, speaker_id, n_utterances=4, mode="enroll"):
        youtube_ids = self.filter_youtube(speaker_id, n_utterances, mode=mode)
        if len(youtube_ids)==0:
            return None

        idx = random.randrange(len(youtube_ids))
        youtube_id = youtube_ids[idx]
        return youtube_id

    def filter_youtube(self, speaker_id, n_utterances, mode="enroll"):
        if mode=="enroll":
            speaker_path = os.path.join(self.enroll_path, speaker_id)
        else:
            speaker_path = os.path.join(self.verify_path, speaker_id)
        youtube_ids = os.listdir(speaker_path)
        # only get folder with n_utterances
        youtube_ids = [yt for yt in youtube_ids if (yt != ".DS_Store"
                                                    and len(os.listdir(os.path.join(speaker_path, yt))) >= n_utterances)]
        return youtube_ids

    def random_utterances(self, speaker_id, youtube_id, n_utterances, mode="enroll"):
        if mode=="enroll":
            speaker_path = os.path.join(self.enroll_path, speaker_id)
        else:
            speaker_path = os.path.join(self.verify_path, speaker_id)
        youtube_path = os.path.join(speaker_path, youtube_id)
        utter_files = os.listdir(youtube_path)
        utter_files = [u for u in utter_files if u != ".DS_Store"]
        random.shuffle(utter_files)
        n_utterances = min(n_utterances, len(utter_files))
        utterances = utter_files[:n_utterances]
        return utterances

# @anvil.server.callable
def verify_vs_enroll(speaker_config_dict, session_dict):

    start_time = time.time()
    speaker_verification = SpeakerVerification(speaker_config_dict)
    n_utterances = 4
    # random speaker
    # centroid = speaker_verification.enroll(n_utterances=n_utterances)

    speaker_enrolled = session_dict["speaker_enrolled"] #"id10283"
    folder_enrolled = session_dict["folder_enrolled"]

    utter_files = speaker_verification.random_utterances(speaker_id=speaker_enrolled,
                                                         youtube_id=folder_enrolled,
                                                     n_utterances=n_utterances,
                                                     mode="enroll")

    if len(utter_files) < n_utterances:
        error = f"Not enough enrolled utterances: {len(utter_files)} < {n_utterances}"
        return (False, error)

    # utter_files = ['00003.wav', '00010.wav', '00006.wav', '00012.wav']

    centroid = speaker_verification.enroll(speaker_id=speaker_enrolled,
                                           youtube_id=folder_enrolled,
                                           utter_files=utter_files,
                                           n_utterances=n_utterances,
                                           print_mode=False)
    if centroid==None:
        error = f"Centroid could not be computed. Ending enrollment"
        return (False, error)

    # print(centroid)

    enroll_time = time.time()

    speaker_new = session_dict["speaker_new"] # "id10282" # "id0"
    youtube_new = session_dict["youtube_new"] # "37XQxZ5lBD8" # "yt0"
    utter_new = session_dict["utter_new"] # "00007.wav", 8, 9 10
    similarity = speaker_verification.verify_speaker(centroid, speaker_new, youtube_new, utter_new,
                                                     mode="verify", print_mode=False)

    verify_time = time.time()

    ini = "The Speaker Verification system has been trained with:\n"
    ini += "- 1211 speakers: 55% male, 45% female\n"
    ini += "- 148.642 utterances\n\n"
    ini += "Once trained, the enrollment and verification processes can be performed without further training\n\n"

    enroll_str = f"Enrollment time: {round(enroll_time - start_time, 2)} seconds\n"
    verify_str = f"Verification time: {round(verify_time - enroll_time, 2)} seconds\n\n"

    speaker_folder_enrolled = speaker_enrolled + " - " + folder_enrolled
    sim_str = str(round(similarity.numpy()*100, 2))[:5] + "%"

    sim_server_str = f"Similarity of anonymous speaker with enrolled {speaker_folder_enrolled} is {sim_str}\n"
    print(enroll_str, verify_str, sim_server_str)

    speaker_folder_name = session_dict["speaker_folder_name"]
    sim_web_str = f"Similarity of anonymous speaker with enrolled {speaker_folder_name} is {sim_str}\n"
    str_total = ini + enroll_str + verify_str + sim_web_str

    return (True, str_total)

if __name__ == "__main__":
    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    speaker_verification = SpeakerVerification(speaker_config_dict)
    n_utterances = 4

    """
    # random speaker
    # centroid = speaker_verification.enroll(n_utterances=n_utterances)

    sel_speakers = speaker_verification.filter_speakers(n_utterances)
    sel_speakers.sort()
    # print(sel_speakers)

    for speaker_id in sel_speakers:
        sel_youtube = speaker_verification.filter_youtube(speaker_id, n_utterances)
        for youtube_id in sel_youtube:
            utter_files = speaker_verification.random_utterances(speaker_id,
                                                                 youtube_id,
                                                                 n_utterances)

            centroid = speaker_verification.enroll(speaker_id=speaker_id,
                                                   youtube_id=youtube_id,
                                                   utter_files=utter_files,
                                                   n_utterances=n_utterances, print_mode=False)
    """

    session_dict = {"speaker_enrolled": "id10306", "speaker_new": "unknown", "youtube_new": "verify00001",
                    "utter_new": "utter_verify_with_id10306.ogg"}

    (status, similarity) = verify_vs_enroll(speaker_config_dict, session_dict)

# anvil.server.wait_forever()

