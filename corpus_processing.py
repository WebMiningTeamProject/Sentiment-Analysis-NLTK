from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import opinion_lexicon
from nltk.corpus import sentence_polarity
import collections
import random
from nltk.corpus import stopwords

###processes sentence_polarity and opinion_lexicon corpus
class Corpus(object):
    #init methode
    def __init__(self):
        self.x = 'hello'

    #create a bag of words (unordered) from corpus
    def bag_of_words(words):
      return dict([(word, True) for word in words.split()])

    #create bag of words (unordered) from newsarticles
    def bag_of_words2(words):
      #words = str(words)
      return dict([(word, True) for word in words])

    #create list of labeled feature set, with feature being a sentence from sentence_polarity corpus
    def sp_label_feats_from_corpus(feature_detector=bag_of_words):
        #english_stops = set(stopwords.words('english'))
        sentence_polarity.ensure_loaded()
        label_feats = collections.defaultdict(list)
        for label in sentence_polarity.categories():
            t_para = sent_tokenize(sentence_polarity.raw(categories = label))
            #t_para = str(t_para)
            #t_para = [word for word in t_para.split() if word not in english_stops]
            for i in range(len(t_para)):
                feats = feature_detector(t_para[i])
                label_feats[label].append(feats)
        return label_feats

    #create list of labeled feature set, with feature being a sentence from opinion_lexicon corpus
    def ol_label_feats_from_corpus(feature_detector=bag_of_words):
        opinion_lexicon.ensure_loaded()
        label_feats = collections.defaultdict(list)
        label = 'pos'
        t_para = word_tokenize(opinion_lexicon.raw(fileids= 'positive-words.txt'))
        for i in range(len(t_para)):
            feats = feature_detector(t_para[i])
            label_feats[label].append(feats)
        label = 'neg'
        t_para = word_tokenize(opinion_lexicon.raw(fileids= 'negative-words.txt'))
        for i in range(len(t_para)):
            feats = feature_detector(t_para[i])
            label_feats[label].append(feats)
        return label_feats

    #random split labeled features
    def random_split_label_feats(lfeats, split=0.75):
      train_feats = []
      test_feats = []
      for label, feats in lfeats.items():
        train_feats.extend([(feat, label) for feat in feats])
      random.shuffle(train_feats)
      cutoff = int(len(train_feats) * split)
      test_feats = train_feats[cutoff:]
      train_feats = train_feats[:cutoff]
      return train_feats, test_feats

    #random split labeled features
    def x_random_split_label_feats(lfeats, split=0.75):
      #train_feats = collections.defaultdict(list)
      #test_feats = collections.defaultdict(list)
      train_feats = []
      test_feats = []
      for label, feats in lfeats.items():
        train_feats[label].append([(feat, label) for feat in feats])
      random.shuffle(train_feats)
      cutoff = int(len(train_feats) * split)
      #test_feats = dict(list(train_feats.items())[cutoff:])
      #train_feats = dict(train_feats.items()[0:cutoff])
      #test_feats = train_feats[cutoff:]
      #train_feats = train_feats[:cutoff]
      return train_feats, test_feats
