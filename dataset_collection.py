import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.load('horses_or_humans', split='train')
for datum in dataset:
    image, bboxes = datum["image"], datum["bboxes"]
