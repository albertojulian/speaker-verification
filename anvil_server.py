import anvil.server

# import sounddevice as sd
# import soundfile as sf
# import base64
# import numpy as np
import os
import yaml
from test import verify_vs_enroll
import pandas as pd
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

anvil_server_key = os.environ.get('ANVIL_SPEAKER_VERIFICATION_SERVER_KEY')
anvil.server.connect(anvil_server_key)

@anvil.server.callable
def registered_users():

    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    rel_path = speaker_config_dict["rel_path"]
    dsr_enrolled_file = speaker_config_dict["dsr_enrolled_file"]
    dsr_enrolled_path = os.path.join(rel_path, dsr_enrolled_file)
    df = pd.read_csv(dsr_enrolled_path, sep=";")

    enrolled_user_list = []
    for _, row in df.iterrows():
        item0 = row["user_name"] + " - " + row["folder_name"]
        item1 = row["user_id"] + "_" + row["folder_id"]
        enrolled_user_list.append((item0, item1))

    print("Loading enrolled users")
    return enrolled_user_list

@anvil.server.callable
def enroll_utterance(stream, speaker_id, enrollment_id, extension):

    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    rel_path = speaker_config_dict["rel_path"]
    enroll_folder = speaker_config_dict["enroll_folder"]
    speaker_path = os.path.join(rel_path, enroll_folder, speaker_id)
    if not os.path.exists(speaker_path):
        os.mkdir(speaker_path)
        print(f"Creating folder {speaker_path}")

    enrollment_path = os.path.join(speaker_path, enrollment_id)
    if not os.path.exists(enrollment_path):
        os.mkdir(enrollment_path)
        print(f"Creating folder {enrollment_path}")

    utter_names = os.listdir(enrollment_path)
    utter_names = [utter_name for utter_name in utter_names if utter_name != '.DS_Store']

    if len(utter_names)==0: # first audio in the folder
        max_utter_id = 0
    else:
        max_utter_id = max([int(utter_name.split(".")[0]) for utter_name in utter_names])

    utter_new_id = str(max_utter_id + 1).zfill(5)
    utter_new = utter_new_id + "." + extension

    file_path = os.path.join(enrollment_path, utter_new)
    print(f"Saving enrollment utterance {file_path}")

    with open(file_path, "wb") as f:
        f.write(stream.get_bytes())

    return (True, "OK")

@anvil.server.callable
def check_enrolled_speaker(speaker_id, enrollment_id):

    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    rel_path = speaker_config_dict["rel_path"]
    enroll_folder = speaker_config_dict["enroll_folder"]
    speaker_path = os.path.join(rel_path, enroll_folder, speaker_id)
    if not os.path.exists(speaker_path):
        error = f"ERROR: no speaker folder {speaker_path}"
        print(error)
        return (False, error)

    enrollment_path = os.path.join(speaker_path, enrollment_id)
    if not os.path.exists(enrollment_path):
        error = f"ERROR: no enrollment folder {enrollment_path}"
        print(error)
        return (False, error)

    return (True, "OK")

@anvil.server.callable
def verify_speaker(stream, extension, speaker_id, folder_id, speaker_folder_name):

    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    rel_path = speaker_config_dict["rel_path"]
    verify_folder = speaker_config_dict["verify_folder"]
    speaker_new = "unknown"
    youtube_new = "verify00001"
    utter_new = "utter_verify_with_" + speaker_id + "." + extension
    file_path = os.path.join(rel_path, verify_folder, speaker_new, youtube_new, utter_new)
    # print(f"Received {type(stream)} in server") # <class 'anvil._serialise.StreamingMedia'>
    print(f"Saving verification utterance {file_path}")
    with open(file_path, "wb") as f:
        f.write(stream.get_bytes())

    # Proceed with verification and get similarity
    session_dict = {}
    session_dict["speaker_new"] = speaker_new
    session_dict["youtube_new"] = youtube_new
    session_dict["utter_new"] = utter_new

    session_dict["speaker_enrolled"] = speaker_id
    session_dict["folder_enrolled"] = folder_id
    session_dict["speaker_folder_name"] = speaker_folder_name

    (ok, value) = verify_vs_enroll(speaker_config_dict, session_dict)

    if ok==False:
        print(value)

    return (ok, value)


anvil.server.wait_forever()

