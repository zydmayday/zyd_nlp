# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 12:59:17 2016

ngrams.py 

@author: yoshihiko
"""

import nltk
#
def word_bigrams(sents):
    return filter(lambda x: x != ('</s>', '<s>'), 
                  nltk.bigrams(make_sent_words(sents)))

def make_sent_words(sents):
    words = []
    for i in range(len(sents)): words += mod_sent(sents[i])
    return words

def mod_sent(tokens):
    tokens.insert(0, '<s>')
    tokens.append('</s>')
    return tokens

#
brown_sents = nltk.corpus.brown.sents()
brown_bigrams = word_bigrams(brown_sents)
brown_fd = nltk.FreqDist(brown_bigrams)

bb = sorted(brown_fd.items(), key=lambda(x): x[1], reverse=True)

#
brown_cfd = nltk.ConditionalFreqDist(brown_bigrams)

#
from nltk.tokenize import word_tokenize
#
def sentence_bigram_prob(s, cfd=brown_cfd, verbose=True):
    return bigram_prob(mod_sent(word_tokenize(s)), cfd, verbose)

def bigram_prob(tokens, cfd=brown_cfd, verbose=True):
    p = 1.0
    bigrams = nltk.bigrams(tokens)
    for x, y in bigrams:
        b_c = cfd[x][y]
        if cfd[x].N() == 0:
            b_p = 0.0
        else:
            b_p = float(b_c) / cfd[x].N()
        p *= b_p
        if verbose: print x, y, b_c, p
    return p

if __name__ == '__main__':
    input = raw_input('input the search phrase: ')
    print sentence_bigram_prob(input)