import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from pathlib import Path
import pickle
from scipy.ndimage.morphology import binary_dilation
import webrtcvad
import struct

class FeatureExtractor:
    def __init__(self, speaker_config_dict):
        super().__init__()
        self.speaker_config_dict = speaker_config_dict

        self.dataset_ratio = speaker_config_dict["dataset_ratio"]
        self.rel_path = speaker_config_dict["rel_path"]
        self.raw_data_folder = speaker_config_dict["raw_data_folder"]
        self.raw_data_path = os.path.join(self.rel_path, self.raw_data_folder)
        self.proc_data_folder = speaker_config_dict["proc_data_folder"]
        self.proc_data_path = os.path.join(self.rel_path, self.proc_data_folder)
        self.enroll_folder = speaker_config_dict["enroll_folder"]
        self.enroll_path = os.path.join(self.rel_path, self.enroll_folder)
        self.verify_folder = speaker_config_dict["verify_folder"]
        self.verify_path = os.path.join(self.rel_path, self.verify_folder)

        self.partials_n_frames = speaker_config_dict["partials_n_frames"]
        self.mel_n_channels = speaker_config_dict["mel_n_channels"]

        self.sampling_rate_common = speaker_config_dict["sampling_rate_common"]
        self.audio_norm_target_dBFS = speaker_config_dict["audio_norm_target_dBFS"]

        self.vad_window_length = speaker_config_dict["vad_window_length"]
        self.vad_moving_average_width = speaker_config_dict["vad_moving_average_width"]
        self.vad_max_silence_length = speaker_config_dict["vad_max_silence_length"]

        self.mel_window_length = speaker_config_dict["mel_window_length"]
        self.mel_window_step = speaker_config_dict["mel_window_step"]

        self.force_proc = speaker_config_dict["force_proc"]

    def mel_one_test(self):
        speaker_id = "id10003"
        youtube_id = "_JpHD6VnJ3I"
        utter_file = "00002.wav"
        mel_spec_frames = self.mel_spectrogram(speaker_id=speaker_id,
                                       youtube_id=youtube_id,
                                       utter_file=utter_file,
                                       plot=True)

        return mel_spec_frames

    def mel_loop(self):

        if not os.path.exists(self.raw_data_path):
            print(f"END: Not found raw data folder: {self.raw_data_path}")
            return

        if not os.path.exists(self.proc_data_path):
            print(f"Creating processed data folder: {self.proc_data_path}")
            os.mkdir(self.proc_data_path)

        speaker_ids = os.listdir(self.raw_data_path)
        last = int(len(speaker_ids) * self.dataset_ratio)
        print(f"Processing utterances of {last} speakers ({int(self.dataset_ratio * 100)}% of total dataset)")

        speaker_ids = speaker_ids[:last]

        for speaker_id in tqdm(speaker_ids):
            self.speaker_mel_loop(speaker_id)


    def speaker_mel_loop(self, speaker_id):

        if speaker_id == ".DS_Store":
            return

        speaker_proc_data_path = os.path.join(self.proc_data_path, speaker_id)
        if not os.path.isdir(speaker_proc_data_path):
            os.mkdir(speaker_proc_data_path)

         # print(f"speaker_id: {speaker_id}")
        speaker_path = os.path.join(self.raw_data_path, speaker_id)
        youtube_ids = os.listdir(speaker_path)
        for youtube_id in youtube_ids:
            if youtube_id == ".DS_Store":
                continue
            youtube_path = os.path.join(speaker_path, youtube_id)
            utter_files = os.listdir(youtube_path)

            for utter_file in utter_files:
                if utter_file[-4:] not in (".wav", ".ogg"):
                    continue

                mel_file = youtube_id + "_" + utter_file.split(".")[0] + ".npy"
                mel_path = os.path.join(speaker_proc_data_path, mel_file)
                if os.path.exists(mel_path) and self.force_proc == False:
                    continue

                mel_spec_frames = self.mel_spectrogram(speaker_id=speaker_id,
                                                       youtube_id=youtube_id,
                                                       utter_file=utter_file,
                                                       plot=False)

                if mel_spec_frames.shape[0] < self.partials_n_frames:
                    continue

                np.save(mel_path, mel_spec_frames)

    def mel_spectrogram(self, speaker_id="id10001",
                        youtube_id="1zcIwhmdeo4",
                        utter_file="00001.wav",
                        mode="train", window="hann",
                        freq_max_hz=8000, plot=False):
        """
        Takes audio and returns the spectrogram
        :param speaker_id: folder name, corresponding to speaker id
        :param youtube_id: subfolder name, corresponding to a youtube video
        :param utter_file: audio file of utterance
        :param mode: "train" "enroll" "verify"
        :param window: type of window applied in STFT (default: hann)
        :param freq_min_hz: 150 Hz (default)
        :param freq_max_hz: 8000 Hz (default 2000); must be less than sr/2 (Nyquist)
        :return: frames of log mel spectrogram corresponding to the utterance audio file
        """
        if mode=="train":
            utter_path = os.path.join(self.raw_data_path, speaker_id, youtube_id, utter_file)
        elif mode=="enroll":
            utter_path = os.path.join(self.enroll_path, speaker_id, youtube_id, utter_file)
        elif mode=="verify":
            utter_path = os.path.join(self.verify_path, speaker_id, youtube_id, utter_file)
        else:
            print(f"ERROR: {mode=} none of train, enroll, verify")

        if not os.path.exists(utter_path):
            print(f"Can not find utter path: {utter_path}")

        utter, sr_utter = librosa.core.load(utter_path, sr=None)
        # VoxCeleb1: All wav files sampled with sr = 16000
        if sr_utter != self.sampling_rate_common:
            print(f"Original sampling rate of utterance is {sr_utter}. Resampling to {self.sampling_rate_common}")
            utter = librosa.resample(utter, sr_utter, self.sampling_rate_common)
            """
            TODO: save to disk with common sampling rate 
            and (preferably) wav format, though should remove the original (oog?)
            """
        utter = self.normalize_volume(utter, self.audio_norm_target_dBFS, increase_only=True)
        utter = self.trim_long_silences(utter)

        duration = int(utter.shape[0]/self.sampling_rate_common)
        # print(f"Number of samples: {utter.shape[0]}, sample rate: {sr_common}, audio duration: {duration} s.")
        # utter = utter[10000:12000]

        # n_fft: size in samples of the frame in which the STFT is going to be applied
        n_fft = int(self.sampling_rate_common * self.mel_window_length / 1000) # 400 samples, if sr_common=16000 and mel_window_length = 25 ms

        # hop_length: step in samples from one frame to the next
        hop_length = int(self.sampling_rate_common * self.mel_window_step / 1000)

        # mel_n_channels: number of mel spectrogram energy frequency bands; 40 Speaker Encoder
        mel_spec_frames = librosa.feature.melspectrogram(y=utter, sr=self.sampling_rate_common,
                            n_fft=n_fft, hop_length=hop_length,
                            n_mels=self.mel_n_channels, window=window, fmax=freq_max_hz)

        # shape of mel_spec_frames: (mel_n_channels, frames) => transpose !!
        mel_spec_frames = mel_spec_frames.astype(np.float32).T

        """
        QUESTION: Should normalize and or convert to db?
        mel_spec_frames = mel_spec_frames / np.max(mel_spec_frames)
        log_mel_spec_frames = librosa.power_to_db(mel_spec_frames, ref=np.max)
        """

        if plot==True:
            plt.plot(utter)
            plt.show()
            plt.close()

            fig, ax = plt.subplots()
            img = librosa.display.specshow(mel_spec_frames, x_axis='time', y_axis='log',
                                           ax=ax, sr=self.sampling_rate_common)
            fig.colorbar(img, ax=ax, format="%+2.f dB", cmap="inferno")
            ax.set(title='Mel spectrogram display')
            plt.show()
            plt.close()

        return mel_spec_frames

    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
        if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
            return wav
        return wav * (10 ** (dBFS_change / 20))


    def trim_long_silences(self, wav):
        """
        Ensures that segments without voice in the waveform remain no longer than a
        threshold determined by the VAD parameters in params.py.
        :param wav: the raw waveform as a numpy array of floats
        :return: the same waveform with silences trimmed away (length <= original wav length)
        """
        int16_max = (2 ** 15) - 1
        # Compute the voice detection window size
        samples_per_window = (self.vad_window_length * self.sampling_rate_common) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.sampling_rate_common))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.vad_moving_average_width)
        # audio_mask = np.round(audio_mask).astype(np.bool)
        audio_mask = np.round(audio_mask).astype(bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]

    def sliding_windows(self, speaker_id, youtube_id, utter_file, mode="enroll", print_mode=True):
        mel_frames = self.mel_spectrogram(speaker_id=speaker_id,
                                          youtube_id=youtube_id,
                                          utter_file=utter_file,
                                          mode=mode)

        # apply sliding window on the complete utter
        if print_mode==True:
            print(f"Shape of mel utterance: {mel_frames.shape[0]} frames, {mel_frames.shape[1]} mel channels")

        utter_len = mel_frames.shape[0]
        inference_n_frames = self.speaker_config_dict["inference_n_frames"]
        offset_frames = inference_n_frames / 2
        n_sliding_windows = int((utter_len - inference_n_frames) / offset_frames) + 1

        if n_sliding_windows < 4:
            print("too short audio")
        sliding_windows = np.zeros((n_sliding_windows, inference_n_frames, mel_frames.shape[1]))
        if print_mode==True:
            print(f"Sliding window batch size: {sliding_windows.shape }")
        for slice in range(n_sliding_windows):
            start_frame = int(slice * offset_frames)
            end_frame = start_frame + inference_n_frames
            # print(f"start_frame: {start_frame}, end_frame: {end_frame}")
            sliding_windows[slice] = mel_frames[start_frame:end_frame]

        return sliding_windows

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # mel_one_test()
    import yaml
    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # speaker_config_dict["dataset_ratio"] = 0.1
    feature_extractor = FeatureExtractor(speaker_config_dict)
    feature_extractor.mel_loop()

"""
Google Cloud
!pip install webrtcvad
!conda install libsndfile -y
!pip install librosa
import yaml
import webrtcvad
voice_cloning = "/home/ajulian/voice-cloning"
%cd {voice_cloning}
from feature_extraction import FeatureExtractor

!chmod 777 /home/ajulian/VoxCeleb1

speaker_file = "SpeakerEncoder.yaml"
with open(speaker_file) as file:
    speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

speaker_config_dict["rel_path"] = "../VoxCeleb1/"
feature_extractor = FeatureExtractor(speaker_config_dict)
feature_extractor.mel_loop()
"""