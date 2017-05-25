
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentence_polarity
import collections
import random

###processes sentence_polarity corpus, first dict transformation and then split into training and testing
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

    #randomly split labeled features
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

