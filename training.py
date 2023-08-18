# Python
import argparse
import os
import random
import sys

# Third party
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# Tensorflow
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_data(data_path: str, label_key: dict):
    """ Makes sentences from ticket dataframes using relevant information for categorization.
    Shuffles the sentences. One-hot encodes each label for a sentence. 

    Args:
        data_path (str): path to dataframe csv
        label_key (dict): label mapping 

    Returns:
        sentences: list of strings
        labels: list of arrays 
    """

    data_df = pd.read_csv(data_path)
    brief_descriptions = data_df["Brief description"].tolist()
    subjects = data_df["Subject"].tolist()
    sentences = [x + ": " + y for x, y in zip(subjects, brief_descriptions)]
    labels = data_df["Category"].tolist()
    labels = [label_key[x] for x in labels]
    indices= list(range(len(sentences)))
    random.shuffle(indices)
    sentences = [sentences[i] for i in indices]
    labels = [labels[i] for i in indices]
    labels = to_categorical(labels, num_classes=len(label_key))

    return sentences, labels


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_directory_path', type=str, default="/home/ggupta/TicketPro/models/")
    parser.add_argument('--training_data_path', type=str, default="/home/ggupta/TicketPro/data/training_data.csv")
    parser.add_argument('--validation_data_path', type=str, default="/home/ggupta/TicketPro/data/validation_data.csv")

    args = parser.parse_args(argv)

    # Constants
    TRAIN_DATA_PATH = args.training_data_path
    VALIDATION_DATA_PATH = args.validation_data_path
    MODEL_DIR_PATH = args.model_directory_path
    LABELS_KEY = {"Software": 0,
                "Network": 1,
                "Hardware": 2}
    VOCAB_SIZE = 2000
    OOV_TOKEN = "<OOV>"
    MAX_LENGTH=13
    TRUNC_TYPE="post"
    EMBEDDING_DIM = 16
    NUM_EPOCHS=args.epochs
    
    # Get data and labels into lists
    training_sentences, training_labels = preprocess_data(TRAIN_DATA_PATH, LABELS_KEY)
    validation_sentences, validation_labels = preprocess_data(VALIDATION_DATA_PATH, LABELS_KEY)

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences=tokenizer.texts_to_sequences(training_sentences)
    training_sequences=pad_sequences(training_sequences,maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    validation_sequences=tokenizer.texts_to_sequences(validation_sentences)
    validation_sequences=pad_sequences(validation_sequences,maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=training_sequences.shape[1]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Setup the training parameters
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(training_sequences, 
                        training_labels, 
                        epochs=NUM_EPOCHS, 
                        validation_data=(validation_sequences, validation_labels))

    # Save the model
    model.save(MODEL_DIR_PATH)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))