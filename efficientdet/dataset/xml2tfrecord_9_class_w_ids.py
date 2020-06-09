from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shutil import move
import tfrecord_util
import tensorflow._api.v2.compat.v1 as tf
import glob
import logging
import argparse
import numpy as np
import random
from tqdm import tqdm
from lxml import etree
import os
import time
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
This script is to convert pascal-voc data format (the xmls generated by labelImg) annotations to tfrecord.

Usage:
python create_tfrecord.py --img_dir /path/to/img --anno_dir /path/to/anno --save_dir /path/to/save --format img_format --num_shards num_shards
The default img_format is png, num_shards is 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--anno_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--format", type=str, default=".png")
parser.add_argument("--num_shards", type=int, default=10)
args = parser.parse_args()

label_id_mapping = {
        'pedestrian': 1,
        'aid-seated': 2,
        'pushable': 3,
        'pullable': 4,
        'mobility-standing': 5,
        'stroller': 6,
        'wheelchair': 7,
        'cycle': 8,
        'cyclist': 9,
        'rider': 1,
        'wheelchair-user': 1,
    }

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]
    return tfrecords


def read_xml_gtbox_and_label(xml_path):
    root = etree.parse(xml_path)
    img_width = None
    img_height = None
    x_min_list = []
    y_min_list = []
    x_max_list = []
    y_max_list = []
    cat_ids = []
    for child_of_root in root.iter():
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        if child_of_root.tag == 'object':
            label_id = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    if child_item.text in label_id_mapping:
                        label_id = label_id_mapping[child_item.text]
                        cat_ids.append(label_id)
                    #else:
                    #    print(child_item.text)
                if child_item.tag == 'bndbox':
                    if label_id is not None:
                        for node in child_item:
                            if node.tag == 'xmin':
                                x_min_list.append(float(node.text))
                            elif node.tag == 'xmax':
                                x_max_list.append(float(node.text))
                            elif node.tag == 'ymin':
                                y_min_list.append(float(node.text))
                            elif node.tag == 'ymax':
                                y_max_list.append(float(node.text))
    x_min_list = [x_min / img_width for x_min in x_min_list]
    y_min_list = [y_min / img_height for y_min in y_min_list]
    x_max_list = [x_max / img_width for x_max in x_max_list]
    y_max_list = [y_max / img_height for y_max in y_max_list]
    return img_height, img_width, x_min_list, y_min_list, x_max_list, y_max_list, cat_ids


def load_txt_annotations(txt_annotation_path):
    with open(txt_annotation_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip()
                       for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


def read_txt_gtbox_and_label(annotation):
    line = annotation.split()
    image_name = line[0].split('/')[-1]
    bboxes = np.array(
        [list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    #shape [m,9]
    bboxes = np.reshape(bboxes, [-1, 9])
    x_list = bboxes[:, 0:-2:2]
    y_list = bboxes[:, 1:-1:2]
    class_id = (bboxes[:, -1]+1).tolist()
    y_max = (np.max(y_list, axis=1)/2048).tolist()
    y_min = (np.min(y_list, axis=1)/2048).tolist()
    x_max = (np.max(x_list, axis=1)/2448).tolist()
    x_min = (np.min(x_list, axis=1)/2448).tolist()
    return image_name, x_min, y_min, x_max, y_max, class_id


GLOBAL_IMG_ID = 0  # global image id.

def get_image_id(filename):
  """Convert a string to a integer."""
  # Warning: this function is highly specific to pascal filename!!
  # Given filename like '2008_000002', we cannot use id 2008000002 because our
  # code internally will convert the int value to float32 and back to int, which
  # would cause value mismatch int(float32(2008000002)) != int(2008000002).
  # COCO needs int values, here we just use a incremental global_id, but
  # users should customize their own ways to generate filename.
  del filename
  global GLOBAL_IMG_ID
  GLOBAL_IMG_ID += 1
  return GLOBAL_IMG_ID

def create_tf_example(img_height, img_width,
                      box_xmin, box_ymin, box_xmax, box_ymax, category_ids,
                      image_path):
    img_full_path = image_path
    with tf.gfile.GFile(img_full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    if img_height and img_width:
        image_height = img_height
        image_width = img_width
    else:
        with tf.Session() as sess:
            image = tf.image.decode_png(encoded_jpg)
            shape_tuple = image.eval().shape
            image_height = shape_tuple[0]
            image_width = shape_tuple[1]

    img_name = os.path.splitext(os.path.basename(img_full_path))[0]
    image_id = get_image_id(img_name)

    feature_dict = {
        'image/filename': tfrecord_util.bytes_feature(img_name.encode('utf8')),
        'image/source_id': tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        'image/height': tfrecord_util.int64_feature(image_height),
        'image/width': tfrecord_util.int64_feature(image_width),
        'image/encoded': tfrecord_util.bytes_feature(encoded_jpg),
        'image/format': tfrecord_util.bytes_feature('png'.encode('utf8')), }
    xmin = box_xmin
    xmax = box_xmax
    ymin = box_ymin
    ymax = box_ymax
    category_ids = category_ids
    feature_dict.update({
        'image/object/bbox/xmin': tfrecord_util.float_list_feature(xmin),
        'image/object/bbox/xmax': tfrecord_util.float_list_feature(xmax),
        'image/object/bbox/ymin': tfrecord_util.float_list_feature(ymin),
        'image/object/bbox/ymax': tfrecord_util.float_list_feature(ymax),
        'image/object/class/label': tfrecord_util.int64_list_feature(category_ids)
    })
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example, img_name, image_id


def create_tf_record_from_xml(image_path, xml_path, tf_output_path,
                              tf_record_num_shards, img_format):
    logging.info('writing to output path: %s', tf_output_path)
    writers = [tf.python_io.TFRecordWriter(tf_output_path + '-%05d-of-%05d.tfrecord' % (i, tf_record_num_shards))
               for i in range(tf_record_num_shards)]

    xml_filepaths = glob.glob(xml_path + '/*.xml')
    random.shuffle(xml_filepaths)

    name_to_id_dict = {}

    for count, xml in enumerate(tqdm(xml_filepaths)):
        xml = xml.replace('\\', '/')
        img_name = os.path.basename(xml).replace('.xml', img_format)
        img_path = image_path + '/' + img_name
        if not os.path.exists(img_path):
            img_name = os.path.basename(xml).replace('.xml', ".jpg")
            img_path = image_path + '/' + img_name
            if not os.path.exists(img_path):
                img_name = os.path.basename(xml).replace('.xml', ".jpeg")
                img_path = image_path + '/' + img_name
                if not os.path.exists(img_path):
                    print('{} does not exist!'.format(img_path))
                    continue
        img_height, img_width, xmin, ymin, xmax, ymax, category_ids = read_xml_gtbox_and_label(
            xml)
        example, filename, image_id = create_tf_example(
            img_height, img_width, xmin, ymin, xmax, ymax, category_ids, img_path)
        
        name_to_id_dict[filename] = image_id

        writers[count % tf_record_num_shards].write(
            example.SerializeToString())
    
    with open(tf_output_path + "_idmap.json", "w") as jsonfile:
        json.dump(name_to_id_dict, jsonfile)
        


def create_tf_record_from_txt(image_dir_path, txt_path, tf_output_path,
                              tf_record_num_shards):

    logging.info('writing to output path: %s', tf_output_path)
    writers = [tf.python_io.TFRecordWriter(tf_output_path + '-%05d-of-%05d.tfrecord' % (i, tf_record_num_shards))
               for i in range(tf_record_num_shards)]
    annotations = load_txt_annotations(txt_path)
    for count, annotation in enumerate(annotations):
        # to avoid path error in different development platform
        print("****************************")
        image_name, xmin, ymin, xmax, ymax, category_ids = read_txt_gtbox_and_label(
            annotation)
        
        img_path = image_dir_path + '/' + image_name
        if not os.path.exists(img_path):
            print('{} does not exist!'.format(img_path))
            continue
        example, _, _ = create_tf_example(
            None, None, xmin, ymin, xmax, ymax, category_ids, img_path)
        writers[count % tf_record_num_shards].write(
            example.SerializeToString())


def main(_):
    create_tf_record_from_xml(image_path=args.img_dir,
                              xml_path=args.anno_dir,
                              tf_output_path=args.save_dir,
                              tf_record_num_shards=args.num_shards, img_format=args.format)


if __name__ == '__main__':
    tf.app.run(main)
