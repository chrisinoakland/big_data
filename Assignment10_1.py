#
# File: Assignment10_1.py
# Name: Christopher M. Anderson
# Date: 11/08/2020
# Course: DSC650 Big Data
# Week: 10
# Assignment Number: 10.1


# In the first part of the assignment, you will implement basic
# text-preprocessing functions in Python. These functions do not
# need to scale to large text documents and will only need to
# handle small inputs.
#
# a.
#
# Create a tokenize function that splits a sentence
# into words. Ensure that your tokenizer removes
# basic punctuation.
#

import nltk
nltk.download('punkt')
from nltk import word_tokenize


def tokenize(sentence):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    my_str = sentence

    # To take input from the user
    # my_str = input("Enter a string: ")

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    print(no_punct)
    tokens = []
    # tokenize the sentence
    tokens = word_tokenize(sentence)

    return tokens


tokenize(sentence="testing this with a sentence. Or two. Hello. $!")

# b.
#
# Implement an ngram function that splits tokens nto N-grams.


def ngram(tokens, n):
    ngrams = []
    # Create ngrams
    for num in range(0, len(tokens)):
        ngram = ' '.join(tokens[num:num + n])
        ngrams.append(ngram)

    return ngrams


print(ngram("cat, dog", 2))

# c.
#
# Implement an one_hot_encode function to create
# a vector from a numerical vector from a list of tokens.
#
# def one_hot_encode(tokens, num_words):
#     token_index = {}
#     results = ''
#     return results

import numpy as np

samples = ['The dog sat on the mat.', 'The cat ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

# Next, we vectorize our samples.
# We will only consider the first `max_length` words in each sample.
max_length = 10

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results)
