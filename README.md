# Speaker Verification
Implementation of Google's Speaker Verification for Android: enrollment based on "OK Google" repetition

This repository implements Android Speaker Verification paper: [GE2E loss](https://arxiv.org/abs/1710.10467).

[Here is a Video of presentation and demo](https://www.youtube.com/watch?v=sSPnZogKkd8)

There is a SpeakerEncoder.yaml configuration file with the parameters of the different stages, which is loaded as a python dictionary.

## Dataset loading

The implementation described in the paper used several datasets: VoxCeleb1, VoxCeleb2 and a Google internal dataset.

This version of the Speaker Encoder has been trained just with the dataset VoxCeleb1, which must be downloaded from:

http://www.robots.ox.ac.uk/%7Evgg/data/voxceleb/vox1.html
