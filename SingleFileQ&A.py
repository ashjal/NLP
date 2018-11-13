###
# Exploring tensor hub model of 'Universal Sentence Encoder' developed by Google.
# Research paper: https://arxiv.org/pdf/1803.11175.pdf
# The brain behind Google TalkToBooks.
# Motivation: To build a Q&A system.
###

# Below line can be used in a Jupyter notebook to enable intellisense (TAB after '.')
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import sys,os,os.path
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import unicodedata

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Import the Universal Sentence Encoder's TF Hub module
# Downloads the whole model (~ 1 GB) when run the first time
# Then onwards, loads from cache (the whole model is saved on machine)
embed = hub.Module(module_url)

path = ""
raw = open(path, 'r')
text = raw.read()
data = unicodedata.normalize("NFKD", text)
raw.close()
data = data.replace('\n', ' ')
data = data.replace('\r', ' ')
data = data.replace('. ', '.')
sentences = data.split(".")
print(sentences[:5])

# Converting text (each sentence) to embedding (1-D vector)
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    sentence_embeddings = session.run(embed(sentences))

print("No. of sentences: ", len(sentences))

print("No. of embeddings: ", len(sentence_embeddings))

print("Embeddings: \n", sentence_embeddings[:5])

print("Dimension of embedding (vector): ", len(sentence_embeddings[0]))

# Your question goes below
question = ['']

# question to embedding
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    question_embeddings = session.run(embed(question))

def most_similar_index(ques_emb, sent_emb):
    index = -1
    similarity = 0
    
    for i, sent in enumerate(np.array(sent_emb).tolist()):
        # inner product of embeddings to calculate similarity index
        score = np.inner(sent, ques_emb)
        if score > similarity:
            similarity = score
            index = i
    
    return index, similarity

index, similarity = most_similar_index(question_embeddings, sentence_embeddings)
print("Answer: ", sentences[index])
print("Similarity score = ", similarity)