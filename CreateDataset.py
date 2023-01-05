import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras import Input
from keras.utils import plot_model
import time
#import statistics_tools as stat
#import pydot

## Declaring constants and functions

data_size = 1024
# num_samples = 4395
# num_ver_samples = 94
# path_to_data = 'C:/Users/vojta/Documents/GitHub/StrojoveUceni/input_data/*/*.csv'
# path_to_ver_data = 'C:/Users/vojta/Documents/GitHub/StrojoveUceni/verification_input_data/*/*.csv'

# For Lko model
num_samples = 220
path_to_data = 'C:/Users/vojta/Documents/GitHub/AnalyzaDatAVypocetniInteligence/input_data_Lko/*/*.csv'


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1000
train_size = int(num_samples * 2 * 0.9)


def get_label(file_path):
    """
    :param file_path: Path to the file with data
    :return: Label of data (broken, unbroken, reference) = (1, 0, 2)
    """
    import os
    if tf.strings.split(file_path, os.path.sep)[-2] == 'broken':
        return tf.constant(1, dtype=tf.int32)
    elif tf.strings.split(file_path, os.path.sep)[-2] == 'unbroken':
        return tf.constant(0, dtype=tf.int32)
    elif tf.strings.split(file_path, os.path.sep)[-2] == 'reference':
        return tf.constant(2, dtype=tf.int32)
    else:
        print(file_path)
        print(tf.strings.split(file_path, os.path.sep))
        print('ERROR FINDING LABEL')
        return tf.constant(0, dtype=tf.int32)


def get_reference_path(tested_path):
    """
    :param tested_path: Path to the file whose status DNN should guess
    :return: Path to the reference path
    """
    ref_path = tf.strings.regex_replace(tested_path, "broken|unbroken", "reference")
    return ref_path


def read_file(path):
    """
    :param path: Path to the file with data
    :return: Dataseries
    """
    file = tf.io.read_file(path)
    file = tf.strings.split(file, sep='\r', maxsplit=-1, name=None)[0]

    dataseries = tf.io.decode_csv(
        file,
        [float()] * data_size,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        select_cols=None,
        name=None
    )

    dataseries = tf.stack(dataseries, axis=0, name='stack')

    return dataseries


def map_files(tested_path):
    """
    :param tested_path: Path to the file whose status DNN should guess
    :return: Input for DNN [tested_dataseries, reference_dataseries], label
    """
    ref_path = get_reference_path(tested_path)

    tested_dataseries = read_file(tested_path)
    reference_dataseries = read_file(ref_path)
    label = get_label(tested_path)

    return [tested_dataseries, reference_dataseries], label


## Create dataset

tim = time.time()
print('creating dataset started')
print(time.time() - tim)
tim = time.time()

dataset = tf.data.Dataset.list_files(path_to_data, shuffle=20)
dataset = dataset.filter(lambda x: get_label(x) < 2)

train_ds = dataset.take(train_size)

train_ds = train_ds.filter(lambda x: get_label(x) < 2)
train_ds = train_ds.map(map_files, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
# train_ds = train_ds.shuffle(train_size)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

# ver_ds = tf.data.Dataset.list_files(path_to_ver_data, shuffle=20)
# ver_ds = ver_ds.filter(lambda x: get_label(x) < 2)
# ver_ds = ver_ds.map(map_files, num_parallel_calls=AUTOTUNE)
# ver_ds = ver_ds.cache()
# ver_ds = ver_ds.batch(BATCH_SIZE)
# ver_ds = ver_ds.prefetch(AUTOTUNE)

## Plot some train data

plot_ds = train_ds.take(1)
fig, (axes) = plt.subplots(3, 2)
for row in plot_ds:
    data = row[0]
    label = row[1]

i = 0
for axes_hor in axes:
    for ax in axes_hor:
        ax.plot(np.arange(0, data_size), data[i, :][0, :])
        ax.plot(np.arange(0, data_size), data[i, :][1, :])
        if label[i] == 1:
            textlabel = 'broken'
        else:
            textlabel = 'unbroken'
        ax.title.set_text(f'train_data, label = {label[i]},  ' + textlabel)
        i += 1

## Define function with output
def ReturnDataset():
    """

    :return: Take Dataset
    """
    return train_ds.take(1)
