import os
from chop_group.speaker_verification_dataset import SpeakerVerificationDataset, SpeakerVerificationDataLoader
import tensorflow as tf
from GE2ELoss import SpeakerEncoder
import matplotlib.pyplot as plt

def train_speaker_verification(speaker_config_dict):
    n_utters = 0
    # speakers_file = "speakers_dict.pkl"
    # speakers_dict = pickle.load(open(speakers_file, "rb"))

    rel_path = speaker_config_dict["rel_path"]
    proc_data_folder = speaker_config_dict["proc_data_folder"]
    proc_data_path = os.path.join(rel_path, proc_data_folder)
    if not os.path.exists(proc_data_path):
        print(f"END: Not found processed data folder: {proc_data_path}")
        return

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=speaker_config_dict["learning_rate"])

    history = model.train(
        loader,
        optimizer,
        epochs=1
    )

    plt.plot(history["train_loss"], label="train_loss")
    plt.show()
    plt.close()

if __name__ == "__main__":
    import yaml
    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    speaker_config_dict["n_dim_out"] = 256
    speaker_config_dict["n_nodes"] = 256 # 512
    speaker_config_dict["max_steps"] = 3000

    # print(speaker_config_dict)
    # from feature_extraction import FeatureExtractor
    # speaker_config_dict["dataset_ratio"] = 0.01
    # feature_extractor = FeatureExtractor(speaker_config_dict)
    # feature_extractor.mel_loop()
    train_speaker_verification(speaker_config_dict)
