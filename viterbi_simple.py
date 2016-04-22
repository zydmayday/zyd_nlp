# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 12:59:17 2016

viterbi_simple.py 

@author: yoshihiko
"""
# You have to be very careful! Subject to BUGS!!

### Making nltk.ConditionalFreqDist tables
### (1) t_w: count(word|tag)
### (2) t_t: count(previous_tag|current_tag)
### as well as pos_tags that maintains the set of pos tags

import nltk
# from here almost borrowed from bigrams.py
def make_tagged_word_bigrams(sents):
    return filter(lambda x: x != (('_end', '</s>'), ('start_', '<s>')), 
                  nltk.bigrams(make_sent_words(sents)))

def make_sent_words(sents):
    words = []
    for i in range(len(sents)): words += mod_sent(sents[i])
    return words

def mod_sent(tokens):
    tokens.insert(0, ('start_', '<s>'))
    tokens.append(('_end', '</s>'))
    return tokens
# to here

# first, read from the nltk treebank corpus
tagged_sents = nltk.corpus.treebank.tagged_sents()
# then, make a list for (word, tag) bigrams
tagged_word_bigrams = make_tagged_word_bigrams(tagged_sents)

# word emission count
t_w = nltk.ConditionalFreqDist([(d[0][1], d[0][0]) for d in tagged_word_bigrams])
# state transition count
t_t = nltk.ConditionalFreqDist([(d[1][1], d[0][1]) for d in tagged_word_bigrams])

# set of pos_tags seen in the corpus
pos_tags = t_t.keys()
S = len(pos_tags)

## Helper funcitons for retrieving probabilities for A and B
# p(t|w) with Add-1 smoothing (for B)
def p_t_w(curr_tag, curr_word):
    '''
    cal the prob of current tag and current word.
    make sure the result will not be zero
    '''
    # num = t_w[curr_tag][curr_word] + 1.0
    num = t_w[curr_tag][curr_word] * 1.0
    denom = t_w[curr_tag].N() + t_w[curr_tag].B()
    return num / denom

# p(t_i|t_i-1) with Add-1 smoothing (for A)
def p_t_t(curr_tag, prev_tag):
    # num = t_t[curr_tag][prev_tag] + 1.0
    num = t_t[curr_tag][prev_tag] * 1.0
    denom = t_t[curr_tag].N() + t_t[prev_tag].B()
    return num / denom

### You have to implement the Viterbi algorithm
def viterbi(tokens): 
    T = len(tokens)
    V = np.zeros((S+2, T+2), dtype=float32)
    B = np.zeros((S+2, T+2))

    # fill your code here
    #
    #

    return tagged_tokens, tokens_prob

def zyd_viterbi(tokens):
    T = len(tokens)
    tagged_tokens = [] * T; tokens_prob = [1] * T
    prev_tag = '<s>'
    def max_tag_prob(prev_tag, curr_word):
        max_tag, max_prob = '', -1
        for t in pos_tags:
            if t != '</s>':
                prob = p_t_t(t, prev_tag) * p_t_w(t, curr_word)
                if prob > max_prob: max_tag, max_prob = t, prob
        return max_tag, max_prob
    for i in range(T):
        m_t, m_p = max_tag_prob(prev_tag, tokens[i])
        tagged_tokens.append(m_t)
        tokens_prob[i] = tokens_prob[i-1] * m_p
        prev_tag = m_t
    return tagged_tokens, tokens_prob

### The function for testing (or having a fun)
from nltk.tokenize import word_tokenize
def pos_tag(sentence):
    tokens = word_tokenize(sentence)
    # You might want to preprocess the tokens? I don't know!
    # tagged_sentence, sentence_probability = viterbi(tokens)
    tagged_sentence, sentence_prob = zyd_viterbi(tokens)
    return tagged_sentence, sentence_prob


if __name__ == '__main__':
    input  = raw_input('please input a sentence: ')
    print pos_tag(input)

