'''

Replicate the results of Figure 12 (right) from the paper 
    DEEP DOUBLE DESCENT: WHERE BIGGER MODELS AND MORE DATA HURT, 2019 (https://arxiv.org/pdf/1912.02292.pdf)

Train Convolution Neural Networks of depth 5 with widths of 6, 30, & 128
with datasets of size 5K, 10K, 20K, 30K, 40K, 50K.

The dataset used is Cifar-10 with 20% label Noise.

'''
import tensorflow as tf

from tensorflow.image import random_crop, resize_with_crop_or_pad
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

import time
import numpy as np
import pickle as pkl
import argparse

from models.conv_nets import make_convNet
from utils.train_utils import load_data, timer, inverse_squareroot_lr


def train_convnet_subsample(label_noise_as_int, n_sample_list, width, load_saved_metrics=True):
    """
        Close replication of the function utils.train_utils.train_convnets but is simplified 
        for the use of testing the data wise double descent.
        
        Models are trained with a batch size of 128 for 1000 epochs using SGD with inverse-square root learning rate.

        Parameters
        ----------
        label_noise_as_int: int
            Amount of label noise to add to the Cifar-10 dataset. Pass as an integer value.
        n_sample_list: list[int]
            A list of sample sizes (must be below 50_000) to train the ConvNets on.
        width: int
            The width of the network to train across the different dataset sizes.
        load_saved_metrics: bool
            pick up from cut of point of prior experiment. Loads file from the data matching the current experiment name.
    """

    label_noise = label_noise_as_int / 100

    # load the relevent dataset. Note that the training data is cast to tf.float32 and normalized by 255.
    (x_train, y_train), (x_test, y_test), image_shape = load_data('cifar10', label_noise)

    batch_size = 128
    n_epochs = 1_000

    # store results for later graphing and analysis.
    metrics = {}

    # Path to save results from training with different sample sizes
    data_save_path = f"subsample_results/width_{width}.pkl"
    
     # load data from prior runs of related experiment.
    if load_saved_metrics:
        try:
            with open(data_save_path, 'rb') as f:
                metrics = pkl.load(f)
        except Exception as e:
            print('Could not find saved metrics.pkl file, exiting')
            raise e

        loaded_sample_sizes = [int(i.split('_')[-1]) for i in metrics.keys()]
        assert n_sample_list[:len(loaded_sample_sizes)] == loaded_sample_sizes
        print('loaded results for width %s from existing file at %s' %(', '.join([str(i) for i in loaded_sample_sizes]), data_save_path))

        assert data_save_path[-4:] == ".pkl"
        data_backup_path = data_save_path[:-4] + 'backup_w%d_' %loaded_sample_sizes[-1] + time.strftime("%D_%H%M%S").replace('/', '') + ".pkl"
        print('saving existing result.pkl to backup at %s' %data_backup_path)
        pkl.dump(metrics, open(data_backup_path, "wb"))
    
    # train the model for each sample size specified.
    for sample_size in n_sample_list:
        if load_saved_metrics and sample_size in loaded_sample_sizes:
            print(f'Sample Size {sample_size} results already loaded from .pkl file, training skipped')
            continue
            
        # Depth 5 Conv Net using default Kaiming Uniform Initialization.
        conv_net, model_id = make_convNet(
            image_shape, depth=5, init_channels=width, n_classes=10
        )

        conv_net.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=inverse_squareroot_lr()),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model_timer = timer()
        
        # Select a random index to subsample the dataset. 
        # use tf.gather to then perform the subsampling below
        idx = np.random.choice(x_train.shape[0], sample_size)

        print(f"STARTING TRAINING: {model_id}, Sample Size: {sample_size}")
        history = conv_net.fit(
            x=tf.gather(x_train, idx) if sample_size < x_train.shape[0] else x_train,
            y=tf.gather(y_train, idx) if sample_size < x_train.shape[0] else y_train,
            validation_data=(x_test, y_test),
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[model_timer],
        )
        print(f"FINISHED TRAINING: {model_id}")

        # add results to dictionary and store the resulting model weights.
        metrics[model_id + f'_{sample_size}'] = history.history
        pkl.dump(metrics, open(data_save_path, "wb"))

        # clear GPU of prior model to decrease training times.
        tf.keras.backend.clear_session()


    return metrics


if __name__ == "__main__":

    # Keeps GPU from claiming all available VRAM.
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # parse optional command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--widths', nargs='+', type=int, default=None)      # adjust the widths used in the experiment
    parser.add_argument('--sample_sizes', nargs='+', type=int, default=None)      # adjust the widths used in the experiment
    parser.add_argument('--noise', type=int, default=10)                    # adjust the label noise used
    args = parser.parse_args()

    subsample_sizes = [5_000, 10_000, 20_000, 30_000, 40_000, 50_000] if args.widths is None else args.sample_sizes
    model_widths = [6, 30, 128] if args.widths is None else args.widths

    for width in model_widths:
        train_convnet_subsample(args.noise, subsample_sizes, width)
