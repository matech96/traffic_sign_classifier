"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import random
import pandas as pd
import tensorflow as tf

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('tf_models_dir', r'C:\Users\gango\Documents\Projects\tf_models',
                    'Directory of the tensorflow model repository.')
flags.DEFINE_string('labels_path', 'gt.txt', 'Path to file containing annotations')
flags.DEFINE_string('images_dir', 'images', 'Directory of the images with .ppm extension')
flags.DEFINE_float('val_prop', 0.1, 'Proportion of the validation set')
FLAGS = flags.FLAGS
sys.path.append(os.path.join(FLAGS.tf_models_dir, "research/object_detection/utils"))
import dataset_util
from collections import namedtuple


def convert_ppm_to_jpg(images_dir):
    for file in os.listdir(images_dir):
        base, ext = os.path.splitext(file)
        new_file_name = os.path.join(images_dir, base + ".jpg")
        if ext == ".ppm" and not os.path.exists(new_file_name):
            img = Image.open(os.path.join(images_dir, file))
            img.save(new_file_name)


def int_to_class_text(id):
    labesl = ["speed limit 20 (prohibitory)",
              "speed limit 30 (prohibitory)",
              "speed limit 50 (prohibitory)",
              "speed limit 60 (prohibitory)",
              "speed limit 70 (prohibitory)",
              "speed limit 80 (prohibitory)",
              "restriction ends 80 (other)",
              "speed limit 100 (prohibitory)",
              "speed limit 120 (prohibitory)",
              "no overtaking (prohibitory)",
              "no overtaking (trucks) (prohibitory)",
              "priority at next intersection (danger)",
              "priority road (other)",
              "give way (other)",
              "stop (other)",
              "no traffic both ways (prohibitory)",
              "no trucks (prohibitory)",
              "no entry (other)",
              "danger (danger)",
              "bend left (danger)",
              "bend right (danger)",
              "bend (danger)",
              "uneven road (danger)",
              "slippery road (danger)",
              "road narrows (danger)",
              "construction (danger)",
              "traffic signal (danger)",
              "pedestrian crossing (danger)",
              "school crossing (danger)",
              "cycles crossing (danger)",
              "snow (danger)",
              "animals (danger)",
              "restriction ends (other)",
              "go right (mandatory)",
              "go left (mandatory)",
              "go straight (mandatory)",
              "go right or straight (mandatory)",
              "go left or straight (mandatory)",
              "keep right (mandatory)",
              "keep left (mandatory)",
              "roundabout (mandatory)",
              "restriction ends (overtaking) (other)",
              "restriction ends (overtaking (trucks)) (other)"]
    return labesl[id]


def group_by_filename(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    base = os.path.splitext(group.filename)[0]
    with tf.gfile.GFile(os.path.join(path, base + ".jpg"), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(int_to_class_text(row['classID']).encode('utf8'))
        classes.append(row['classID']+1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def group_to_tf_record(images_path, record_name, groups):
    names_list = []
    writer = tf.python_io.TFRecordWriter(record_name)
    for group in groups:
        names_list.append(group.filename)
        tf_example = create_tf_example(group, images_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    names_list.sort()
    with open(record_name + ".namestxt", 'w') as f:
        for name in names_list:
            f.write(name + "\n")
    output_path = os.path.join(os.getcwd(), record_name)
    print('Successfully created the TFRecords: {}'.format(output_path))


def main(_):
    images_dir = FLAGS.images_dir
    convert_ppm_to_jpg(images_dir)

    labels_path = FLAGS.labels_path
    val_prop = FLAGS.val_prop
    df = pd.read_csv(labels_path, sep=';', header=None, names=('filename', 'xmin', 'ymin', 'xmax', 'ymax', 'classID'))
    train_files = []
    val_files = []
    for c in df["classID"].unique():
        files_for_class = df[df["classID"] == c]["filename"].tolist()
        n_examples = len(files_for_class)
        border = max(1, int(n_examples * val_prop))
        train_files = union_list(files_for_class[border:], train_files)
        val_files = union_list(files_for_class[:border], val_files)
    grouped = group_by_filename(df, 'filename')
    random.shuffle(grouped)
    print("Train size: " + str(len(train_files)))
    print("Validation size: " + str(len(val_files)))
    train_set = group_by_filename(df[df["filename"].isin(train_files)], 'filename')
    val_set = group_by_filename(df[df["filename"].isin(val_files)], 'filename')
    group_to_tf_record(images_dir, "train.record", train_set)
    group_to_tf_record(images_dir, "validation.record", val_set)


def union_list(a, b):
    return list(set().union(a, b))


if __name__ == '__main__':
    tf.app.run()
