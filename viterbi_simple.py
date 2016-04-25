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

class Katz():
    def __init__(self, k=5, data=None, total_num=0, total_count=0):
        self.Cs = []
        self.k = k
        self.Ns_dict = {}
        self.data = data
        self.total_num = total_num
        self.total_count = total_count
        if data:
            self.generate_Ns_dict()

    def generate_Ns_dict(self):
        '''
        generate the dict of N
        '''
        data = self.data
        k = self.k
        for t,fd in data.items():
            self.Ns_dict[t] = []
            nr = fd.r_Nr()
            self.Ns_dict[t].append(total_num - sum([nr[i] for i in nr]))
            for i in range(1, k+2):
                self.Ns_dict[t].append(nr[i])


    def get_c(self, i=0, condi=None):
        '''
        get c_star
        '''
        k = self.k
        if i > k:
            return i * 1.0
        Ns = self.Ns_dict[condi]
        c = 0.0
        if Ns[i+1]:
            c = ((i + 1.0) * (Ns[i+1] / (Ns[i] + 1)) - i * ((k + 1) * Ns[k+1] / (Ns[1] + 1))) / ((1 - (k + 1) * Ns[k+1] / (Ns[1] + 1)) + 1)
        else:
            c = 1 / total_count
        return c

    def get_p(self, condi=None, para=None):
        '''
        get prob of para under condi
        '''
        Ns = self.Ns_dict[condi]
        data = self.data
        c = data[condi][para]
        p = 0.0
        if not c:
            p = float(Ns[1]) / Ns[0] / total_num
        else:
            p = self.get_c(c, condi) / total_num
        return p

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

# count the amount of the words

total_num = len(set([j[0] for i in tagged_sents for j in i]))
total_count = len([j[0] for i in tagged_sents for j in i])

# then, make a list for (word, tag) bigrams
tagged_word_bigrams = make_tagged_word_bigrams(tagged_sents)

# word emission count
t_w = nltk.ConditionalFreqDist([(d[0][1], d[0][0]) for d in tagged_word_bigrams])
# state transition count
t_t = nltk.ConditionalFreqDist([(d[0][1], d[1][1]) for d in tagged_word_bigrams])

# set of pos_tags seen in the corpus
pos_tags = t_t.keys()
S = len(pos_tags)

p_katz_t_w = Katz(data=t_w, total_num=total_num, total_count=total_count)
p_katz_t_t = Katz(data=t_t, total_num=total_num, total_count=total_count)

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
            prob = p_katz_t_t.get_p(prev_tag, t) * p_katz_t_w.get_p(t, curr_word)
            print t, prev_tag, curr_word, prob
            if prob > max_prob: max_tag, max_prob = t, prob
        print max_tag, max_prob, '========='
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
    while True:
        input  = raw_input('please input a sentence: ')
        print pos_tag(input)
    

