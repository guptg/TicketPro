# Python
import argparse
import os
import sys
import random

# Third party
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Tensorflow
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])


    args = parser.parse_args(argv)

    # Constants
    TRAIN_DATA_PATH = args.training
    VALIDATION_DATA_PATH = args.validation
    LABELS_KEY = {"Software": 0,
                "Network": 1,
                "Hardware": 2}
    VOCAB_SIZE = 2000
    OOV_TOKEN = "<OOV>"
    EMBEDDING_DIM = 16
    NUM_EPOCHS=args.epochs
    MAX_LENGTH=13
    TRUNC_TYPE="post"
    MODEL_DIR = args.model_dir
    
    # Get data and labels into lists
    data_df = pd.read_csv(TRAIN_DATA_PATH)
    brief_descriptions = data_df["Brief description"].tolist()
    subjects = data_df["Subject"].tolist()
    sentences = [x + ": " + y for x, y in zip(subjects, brief_descriptions)]
    labels = data_df["Category"].tolist()
    labels = [LABELS_KEY[x] for x in labels]
    indices= list(range(len(sentences)))
    random.shuffle(indices)
    train_sentences = [sentences[i] for i in indices]
    labels = [labels[i] for i in indices]
    train_labels = to_categorical(labels, num_classes=len(LABELS_KEY))

    val_data_df = pd.read_csv(VALIDATION_DATA_PATH)
    val_brief_descriptions = val_data_df["Brief description"].tolist()
    val_subjects = val_data_df["Subject"].tolist()
    val_sentences = [x + ": " + y for x, y in zip(val_subjects, val_brief_descriptions)]
    val_labels = val_data_df["Category"].tolist()
    val_labels = [LABELS_KEY[x] for x in val_labels]
    indices= list(range(len(val_sentences)))
    random.shuffle(indices)
    test_sentences = [val_sentences[i] for i in indices]
    val_labels = [val_labels[i] for i in indices]
    test_labels = to_categorical(val_labels, num_classes=len(LABELS_KEY))


    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)
    word_index=tokenizer.word_index

    train_sequences=tokenizer.texts_to_sequences(train_sentences)
    train_padded=pad_sequences(train_sequences,maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    test_sequences=tokenizer.texts_to_sequences(test_sentences)
    test_padded=pad_sequences(test_sequences,maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=train_padded.shape[1]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Setup the training parameters
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model.summary()

    
    # Train the model
    history = model.fit(train_padded, train_labels, epochs=NUM_EPOCHS, validation_data=(test_padded, test_labels))

    # Save the model
    model.save(MODEL_DIR)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))