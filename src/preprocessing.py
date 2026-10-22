import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import json
import random
from PIL import Image
from IPython.display import display

def load_image(image_path):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,(299,299))
    image=tf.keras.applications.inception_v3.preprocess_input(image)
    return image, image_path
image_model=tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input=image_model.input
hidden_layer=image_model.layers[-1].output
image_features_extract_model=tf.keras.Model(new_input,hidden_layer)
encode_train=sorted(set(image_name_vector))
image_dataset=tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset=image_dataset.map(load_image).batch(32)
from tqdm import tqdm
for img, path in tqdm(image_dataset):
    batch_features=image_features_extract_model(img)
    batch_features=tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_features=p.numpy().decode("utf-8")
        np.save(path_of_features,bf.numpy())
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
top_k=5000
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>']=0
tokenizer.index_word[0]='<pad>'
train_seqs=tokenizer.texts_to_sequences(train_captions)
cap_vector=tf.keras.preprocessing.sequence.pad_sequences(train_seqs,padding="post")
max_length=calc_max_length(train_seqs)
max_length
