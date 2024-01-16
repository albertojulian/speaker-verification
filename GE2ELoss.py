import numpy as np
import tensorflow as tf
#from tensorflow.keras import models, layers, Model
#import tensorflow
#from tensorflow import keras
from keras import models, layers, Model

import pickle
import os
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
# from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from datetime import datetime

class SpeakerEncoder(tf.keras.Model):
    def __init__(self, speaker_config_dict, train_mode=True):
        super().__init__()
        # super(SpeakerEncoder, self).__init__()
        self.speaker_config_dict = speaker_config_dict

        self.train_mode = train_mode
        if train_mode==True:
            n_frames = speaker_config_dict["partials_n_frames"]
        else:
            n_frames = speaker_config_dict["inference_n_frames"]

        mel_n_channels = speaker_config_dict["mel_n_channels"]

        n_nodes = speaker_config_dict["n_nodes"]
        n_dim_out = speaker_config_dict["n_dim_out"]

        # self.prepro = layers.Reshape((n_frames, mel_n_channels), input_shape=(n_frames, mel_n_channels, 1))
        # self.swap = layers.Permute((2, 1), input_shape=(n_frames, mel_n_channels))
        # self.prepro = layers.Reshape((mel_n_channels, n_frames), input_shape=(mel_n_channels, n_frames, 1))
        self.lstm1 = layers.LSTM(n_nodes, return_sequences=True, input_shape=(n_frames, mel_n_channels))
        self.lstm2 = layers.LSTM(n_nodes, return_sequences=True)
        self.lstm3 = layers.LSTM(n_nodes)
        self.linear = layers.Dense(n_dim_out) # embeddings here
        # self.activ = layers.ReLU()

        similarity_weight = speaker_config_dict["similarity_weight"]
        similarity_bias = speaker_config_dict["similarity_bias"]

        self.similarity_weight = tf.Variable([similarity_weight], dtype=tf.float32)
        self.similarity_bias = tf.Variable([similarity_bias], dtype=tf.float32)

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.save_mode = speaker_config_dict["save_mode"]
        self.load_mode = speaker_config_dict["load_mode"]

        self.checkpoint_path = speaker_config_dict["checkpoint_path"]
        self.optimizer = None

    def call(self, x, training=False, mask=None):
        # x is tensorflow.python.framework.ops.Tensor
        # batch_size = speakers_per_batch * utterances_per_speaker; 16 * 8 = 128
        # x = self.swap(x)
        # x = self.prepro(x)
        # print(x.shape) # (batch_size, partials_n_frames, mel_n_channels): (128, 160, 40)
        x = self.lstm1(x)
        # print(x.shape) # (batch_size, partials_n_frames, n_nodes): (128, 160, 256)
        x = self.lstm2(x)
        # print(x.shape) # (batch_size, partials_n_frames, n_nodes): (128, 160, 256)
        x = self.lstm3(x)
        # print(x.shape) # (batch_size, n_nodes): (128, 256)
        x = self.linear(x)
        # print(x.shape) # (batch_size, n_dim_out): (128, 128)
        # embeds_raw = self.activ(x) # ReLU
        # embeds_raw = tf.keras.activations.sigmoid(x)
        embeds_raw = x

        """
        tf.keras.utils.normalize expects numpy rather than Tensor, sometimes crashes the graph
        """
        # embeds = tf.keras.utils.normalize(embeds_raw, axis=1, order=2)
        embeds  = self.normalize(embeds_raw)

        return embeds

    def normalize(self, x):
        """ normalize the last dimension vector of the input matrix
        :return: normalized input
        """
        return x / tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keepdims=True) + 1e-6)

    def train(self, loader, optimizer):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_loss_debug": [],
            "validate_loss": [],
            "validate_accuracy": []
        }

        self.optimizer = optimizer

        repeat_batch = self.speaker_config_dict["repeat_batch"]
        print_every_n_step = self.speaker_config_dict["print_every_n_step"]
        save_every_n_step = self.speaker_config_dict["save_every_n_step"]

        # for epoch in range(1, epochs + 1):
        mean_train_loss = tf.keras.metrics.Mean()
        max_session_steps = self.speaker_config_dict["max_session_steps"]
        init_step = self.speaker_config_dict["current_step"]
        max_steps = init_step + max_session_steps

        current_session_step = 1
        for step, speaker_batch in enumerate(loader, init_step):
            if repeat_batch==False or (repeat_batch==True and current_session_step==1):
                batch_data = speaker_batch.get_data() # numpy.ndarray (80, 160, 40)
                # print("type(batch_data), shape", type(batch_data), batch_data.shape)
                batch_data = tf.convert_to_tensor(batch_data) # EagerTensor [ 80 160  40]
            # print("type(batch_data), shape", type(batch_data), tf.shape(batch_data))

            # Do a training step.
            # Model.train_step returns a dict, no matter the mode
            if self.speaker_config_dict["eager_tensor_mode"]==True or self.train_mode==False:
                loss_dict = self.train_step(batch_data)
            else:
                loss_dict = train_step(self, batch_data)

            loss_value = loss_dict["loss"]
            sim_matrix = loss_dict["sim_matrix"]

            # avoids gradients and optional non-eager mode (@tf.function; old TF1 graph?)
            eer = self.eer(sim_matrix)
            # eer2 = self.eer2(sim_matrix)

            if step % print_every_n_step == 0:
                print(f"step: {step} loss: {loss_value} eer: {eer}")
                # print(f"step: {step} loss: {loss_value} eer: {eer} eer2: {eer2}")

            if (step % save_every_n_step) == 0 and self.save_mode==True:
                self.speaker_config_dict["similarity_weight"] = self.similarity_weight.numpy()[0]
                self.speaker_config_dict["similarity_bias"] = self.similarity_bias.numpy()[0]

                """
                learning_rate = optimizer.learning_rate.numpy()
                beta_1 = optimizer.beta_1.numpy()
                beta_2 = optimizer.beta_2.numpy()
                """
                learning_rate = self.optimizer.learning_rate.numpy()
                beta_1 = self.optimizer.beta_1.numpy()
                beta_2 = self.optimizer.beta_2.numpy()
                print(f"OPTIMIZER lr: {learning_rate}, b1: {beta_1}, b2: {beta_2}")
                self.speaker_config_dict["learning_rate"] = learning_rate
                self.speaker_config_dict["beta_1"] = beta_1
                self.speaker_config_dict["beta_2"] = beta_2

                # + 1 to avoid saving at the first step when loading in new session
                self.speaker_config_dict["current_step"] = step + 1

                if not os.path.exists(self.checkpoint_path):
                    print(f"Creating model checkpoint folder: {self.checkpoint_path}")
                    os.mkdir(self.checkpoint_path)

                speaker_config_dict_pkl = self.speaker_config_dict["speaker_config_dict_pkl"]
                speaker_config_dict_path = os.path.join(self.checkpoint_path, speaker_config_dict_pkl)
                pickle.dump(self.speaker_config_dict, open(speaker_config_dict_path, "wb"))
                print("Saving dictionary")

                # saved_model_path
                speaker_checkpoint = self.speaker_config_dict["speaker_model_checkpoint"]
                model_path = os.path.join(self.checkpoint_path, speaker_checkpoint)
                self.save_weights(model_path)
                print("Saving model weights")

                checkpoints_bk_path = self.speaker_config_dict["checkpoints_bk_path"]
                if not os.path.exists(checkpoints_bk_path):
                    print(f"Not found checkpoints backup folder: {checkpoints_bk_path}")
                    os.mkdir(checkpoints_bk_path)

                now = datetime.now()
                checkpoint_bk_folder = self.checkpoint_path + "_D" + now.strftime("%d_%H_%M") + "_" + str(step)
                checkpoint_bk_path = os.path.join(checkpoints_bk_path, checkpoint_bk_folder)
                if not os.path.exists(checkpoint_bk_path):
                    print(f"Creating model checkpoint BK folder: {checkpoint_bk_path}")
                    os.mkdir(checkpoint_bk_path)

                speaker_config_dict_bk_path = os.path.join(checkpoint_bk_path, speaker_config_dict_pkl)
                pickle.dump(self.speaker_config_dict, open(speaker_config_dict_bk_path, "wb"))

                model_bk_path = os.path.join(checkpoint_bk_path, speaker_checkpoint)
                self.save_weights(model_bk_path)

                """
                TODO: 
                https://www.tensorflow.org/tutorials/keras/save_and_load?hl=en
                Saving model rather than just weights gives error:
                NotImplementedError: Cannot convert a symbolic Tensor (speaker_encoder/lstm/strided_slice:0)
                to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call,
                which is not supported
                In Colab it gives just a warning:
                WARNING:absl:Found untraced functions such as 
                lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, 
                lstm_cell_1_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses, 
                lstm_cell_2_layer_call_fn while saving (showing 5 of 15). 
                These functions will not be directly callable after loading.
                """
                # tf.keras.models.save_model(self, model_path)
                # self.save(model_path)
                # print("Saving model")

            mean_train_loss(loss_value)

            train_loss = mean_train_loss.result()
            history["train_loss"] += [train_loss] # append loss value for current epoch

            if step==max_steps:
                # print("self.similarity_weight", self.similarity_weight)
                # print("self.similarity_bias", self.similarity_bias)
                return history

            current_session_step += 1

    def train_step(self, x):

        speakers_per_batch = self.speaker_config_dict["speakers_per_batch"]
        utterances_per_speaker = self.speaker_config_dict["utterances_per_speaker"]
        n_dim_out = self.speaker_config_dict["n_dim_out"]

        with tf.GradientTape() as tape:
            embeds = self.call(x, training=True, mask=None)
            embeds = tf.reshape(embeds, (speakers_per_batch, utterances_per_speaker, n_dim_out))

            loss, sim_matrix = self.compute_loss(embeds, speakers_per_batch, utterances_per_speaker)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss, "sim_matrix": sim_matrix} # Model.train_step returns a dictionary

    def compute_loss(self, embeds, speakers_per_batch, utterances_per_speaker):
        # Softmax

        # calculate similarity matrix from centroids and embeddings
        sim_matrix = self.similarity(embeds, speakers_per_batch, utterances_per_speaker)

        # Now we would reshape from N, M, N to NxM, N, but is already done
        # sim_matrix = tf.reshape(sim_matrix, (speakers_per_batch * utterances_per_speaker,
        #                               speakers_per_batch))

        target = tf.repeat(tf.range(speakers_per_batch, dtype=tf.float32), utterances_per_speaker)

        loss = self.loss_fn(target, sim_matrix)
        # with reduce_mean, loss (Keras) = loss2 (TF)
        # loss2 = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, sim_matrix, from_logits=True))

        return loss, sim_matrix

    def similarity(self, embedded_split, speakers_per_batch, utterances_per_speaker, center=None):
        N = speakers_per_batch
        M = utterances_per_speaker
        P = self.speaker_config_dict["n_dim_out"]

        if center is None:
            center = self.normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
            # center = tf.keras.utils.normalize(tf.reduce_mean(embedded_split, axis=1), axis=1, order=2)
            center_except = self.normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True)
                                                 - embedded_split, shape=[N*M, P]))  # [NM,P] center vectors eq.(8)
            # center_except = tf.keras.utils.normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True)
            #                                     - embedded_split, shape=[N*M, P]) , axis=1, order=2)  # [NM,P] center vectors eq.(8)
            # make similarity matrix eq.(9)
            S = tf.concat(
                [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keepdims=True) if i==j
                            else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keepdims=True) for i in range(N)],
                           axis=1) for j in range(N)], axis=0)

            # next not needed in enrollment, only in training
            S = tf.abs(self.similarity_weight) * S + self.similarity_bias  # rescaling

        else :
            # If center(enrollment) exist, use it.
            S = tf.concat(
                [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keepdims=True) for i
                            in range(N)],
                           axis=1) for j in range(N)], axis=0)

        return S

    def eer(self, sim_matrix):
        speakers_per_batch = self.speaker_config_dict["speakers_per_batch"]
        utterances_per_speaker = self.speaker_config_dict["utterances_per_speaker"]

        inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int32)[0]
        ground_truth = np.repeat(np.arange(speakers_per_batch, dtype=np.int32), utterances_per_speaker)
        labels = np.array([inv_argmax(i) for i in ground_truth])
        preds = sim_matrix.numpy()

        # Snippet from https://yangcha.github.io/EER-ROC/
        fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer

    def eer2(self, sim_matrix):
        speakers_per_batch = self.speaker_config_dict["speakers_per_batch"]
        utterances_per_speaker = self.speaker_config_dict["utterances_per_speaker"]

        diff = 1; EER=0; # EER_thres = 0; EER_FAR=0; EER_FRR=0

        sim_matrix = tf.reshape(sim_matrix, (speakers_per_batch, utterances_per_speaker,
                          speakers_per_batch))

        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        for thres in [0.01*i+0.5 for i in range(50)]:
            S_thres = sim_matrix.numpy() > thres

            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != verification speaker)
            FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(speakers_per_batch)])/(speakers_per_batch-1)/utterances_per_speaker/speakers_per_batch

            # False reject ratio = false reject / matched population (enroll speaker = verification speaker)
            FRR = sum([utterances_per_speaker-np.sum(S_thres[i][:,i]) for i in range(speakers_per_batch)])/utterances_per_speaker/speakers_per_batch

            # Save threshold when FAR = FRR (=EER)
            if diff> abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                # EER_thres = thres
                # EER_FAR = FAR
                # EER_FRR = FRR

        return EER

    def get_speaker_embedding(self, sliding_windows):
        checkpoint_path = self.speaker_config_dict["checkpoint_path"]
        speaker_checkpoint = self.speaker_config_dict["speaker_model_checkpoint"]
        model_path = os.path.join(checkpoint_path, speaker_checkpoint)
        # print(model_path)
        # load saved weights; expect_partial avoids warnings of not using optimizer weights
        self.load_weights(model_path).expect_partial()

        embeddings = self.call(tf.convert_to_tensor(sliding_windows), training=False)
        # ALT 1: average embedding in TF
        # embedding_mean = tf.reduce_mean(embeddings, axis=0)
        embedding_mean = self.normalize(tf.reduce_mean(embeddings, axis=0))
        # print(f"Embedding size: {embedding_mean.shape[0]} dimensions")

        # ALT 2: average embedding in numpy
        # embedding_mean = np.mean(embeddings.numpy(), axis=0)
        # print(f"Embedding size: {embedding_mean.shape[0]} dimensions")

        return embedding_mean

    def get_centroid(self, embedding_batch):
        centroid = self.normalize(tf.reduce_mean(embedding_batch, axis=0))
        return centroid

# @tf.autograph.experimental.do_not_convert
@tf.function
def train_step(model, x):

    speakers_per_batch = model.speaker_config_dict["speakers_per_batch"]
    utterances_per_speaker = model.speaker_config_dict["utterances_per_speaker"]
    n_dim_out = model.speaker_config_dict["n_dim_out"]

    with tf.GradientTape() as tape:
        embeds = model.call(x, training=True, mask=None)
        embeds = tf.reshape(embeds, (speakers_per_batch, utterances_per_speaker, n_dim_out))

        # loss, eer = self.compute_loss(embeds, speakers_per_batch, utterances_per_speaker)
        loss, sim_matrix = compute_loss(model, embeds, speakers_per_batch, utterances_per_speaker)

    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {"loss": loss, "sim_matrix": sim_matrix} # Model.train_step returns a dictionary

@tf.function
def compute_loss(model, embeds, speakers_per_batch, utterances_per_speaker):
    # Softmax

    # calculate similarity matrix from centroids and embeddings
    sim_matrix = similarity(model, embeds, speakers_per_batch, utterances_per_speaker)

    # Now we would reshape from N, M, N to NxM, N, but is already done
    # sim_matrix = tf.reshape(sim_matrix, (speakers_per_batch * utterances_per_speaker,
    #                               speakers_per_batch))

    target = tf.repeat(tf.range(speakers_per_batch, dtype=tf.float32), utterances_per_speaker)

    loss = model.loss_fn(target, sim_matrix)
    # with reduce_mean, loss (Keras) = loss2 (TF)
    # loss2 = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, sim_matrix, from_logits=True))

    return loss, sim_matrix

@tf.function
def similarity(model, embedded_split, speakers_per_batch, utterances_per_speaker, center=None):
    N = speakers_per_batch
    M = utterances_per_speaker
    P = model.speaker_config_dict["n_dim_out"]

    if center is None:
        center = model.normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        # center = tf.keras.utils.normalize(tf.reduce_mean(embedded_split, axis=1), axis=1, order=2)
        center_except = model.normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True)
                                             - embedded_split, shape=[N*M, P]))  # [NM,P] center vectors eq.(8)
        # center_except = tf.keras.utils.normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True)
        #                                     - embedded_split, shape=[N*M, P]) , axis=1, order=2)  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keepdims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keepdims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exists, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(model.similarity_weight) * S + model.similarity_bias   # rescaling

    return S


if __name__ == "__main__":
    import yaml
    speaker_file = "SpeakerEncoder.yaml"
    with open(speaker_file) as file:
       speaker_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    se = SpeakerEncoder(speaker_config_dict)
    print(se.similarity_bias.numpy())