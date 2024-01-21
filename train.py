import os
from chop_group.speaker_verification_dataset import SpeakerVerificationDataset, SpeakerVerificationDataLoader
import tensorflow as tf
from GE2ELoss import SpeakerEncoder
import matplotlib.pyplot as plt
import yaml
import pickle
import time

def train_speaker_verification(speaker_config_dict):
    t_start = time.time()

    rel_path = speaker_config_dict["rel_path"]
    proc_data_folder = speaker_config_dict["proc_data_folder"]
    proc_data_path = os.path.join(rel_path, proc_data_folder)
    if not os.path.exists(proc_data_path):
        print(f"END: Not found processed data folder: {proc_data_path}")
        return

    load_mode = speaker_config_dict["load_mode"]
    checkpoint_path = speaker_config_dict["checkpoint_path"]
    if os.path.exists(checkpoint_path) and load_mode==True:
        print(f"Found model checkpoint folder: {checkpoint_path}")

        speaker_config_dict_pkl = speaker_config_dict["speaker_config_dict_pkl"]
        speaker_config_dict_path = os.path.join(checkpoint_path, speaker_config_dict_pkl)
        if os.path.exists(speaker_config_dict_path):
            # speaker_config_dict is replaced
            speaker_config_dict = pickle.load(open(speaker_config_dict_path, "rb"))

    speakers_list = os.listdir(proc_data_path)
    speakers_list = [s for s in speakers_list if s != ".DS_Store"] # MacOS adds those files

    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(speakers_list, proc_data_path)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speaker_config_dict,
        num_workers=0,
    )

    model = SpeakerEncoder(speaker_config_dict)


    """
    n_frames = speaker_config_dict["partials_n_frames"]
    mel_n_channels = speaker_config_dict["mel_n_channels"]
    model.build(input_shape = (n_frames, mel_n_channels, 1))
    print(model.summary())
    """


    if os.path.exists(checkpoint_path) and load_mode==True:
        speaker_checkpoint = speaker_config_dict["speaker_model_checkpoint"]
        model_path = os.path.join(checkpoint_path, speaker_checkpoint)
        print("Loading model checkpoint: just weights")
        # load (previously saved) weights
        model.load_weights(model_path)

        """
        In Google Colab the model was saved with warnings (see GE2ELoss);
        here it gives a warning:
        WARNING:tensorflow:No training configuration found in save file, 
        so the model was *not* compiled. Compile it manually.
        """
        # print("Loading model checkpoint: complete")
        # model = tf.keras.models.load_model(model_path)
        # model.speaker_config_dict = speaker_config_dict

    learning_rate = speaker_config_dict["learning_rate"]
    beta_1 = speaker_config_dict["beta_1"]
    beta_2 = speaker_config_dict["beta_2"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=beta_1, beta_2=beta_2)

    """
    In Google Colab, when loading a saved model
    AttributeError: 'SpeakerEncoder' object has no attribute 'train'
    """
    history = model.train(
        loader,
        optimizer,
        epochs=1
    )

    t_end = time.time()

    max_session_steps = speaker_config_dict["max_session_steps"]
    print(f"Training {max_session_steps} steps takes {(t_end - t_start)}")
    # 50 steps: 17 sec; Colab WITH @tf.function 1000 in 40 sec => 1M in 11 hours

    if speaker_config_dict["plot_history"]==True:
        plt.plot(history["train_loss"], label="train_loss")
        plt.show()
        plt.close()

if __name__ == "__main__":
    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # speaker_config_dict["save_every_n_step"] = 50

    speaker_config_dict["max_session_steps"] = 50
    # speaker_config_dict["load_mode"] = False
    speaker_config_dict["eager_tensor_mode"] = False # if False, uses @tf.function in train_step
    # print(speaker_config_dict)
    train_speaker_verification(speaker_config_dict)

"""
Google Cloud
import yaml
!conda install pytorch
voice_cloning = "/home/ajulian/voice-cloning"
%cd {voice_cloning}

from train import train_speaker_verification

speaker_file = "SpeakerEncoder.yaml"
with open(speaker_file) as file:
    speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

speaker_config_dict["rel_path"] = "../VoxCeleb1/"
train_speaker_verification(speaker_config_dict)
"""