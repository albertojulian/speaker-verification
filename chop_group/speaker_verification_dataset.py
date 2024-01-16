from chop_group.random_cycler import RandomCycler
from chop_group.speaker_batch import SpeakerBatch
from chop_group.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# TODO: improve with a pool of speakers for data efficiency

class SpeakerVerificationDataset(Dataset):
    def __init__(self, speakers_list, proc_data_path):
        # self.speakers_dict = speakers_dict
        # speakers_dicts = list(speakers_dict.values())
        self.speakers = [Speaker(speaker_dict_file, proc_data_path) for speaker_dict_file in speakers_list]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    """
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    """
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speaker_config_dict, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = speaker_config_dict["utterances_per_speaker"]
        self.partials_n_frames = speaker_config_dict["partials_n_frames"]

        super().__init__(
            dataset=dataset, 
            batch_size=speaker_config_dict["speakers_per_batch"],
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, self.partials_n_frames)
    