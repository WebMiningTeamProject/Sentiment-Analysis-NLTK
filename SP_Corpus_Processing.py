from nltk.tokenize import sent_tokenize
import collections
import random
from nltk.corpus import sentence_polarity

class SP_Corpus(object):
    #create a bag of words (unordered) from corpus
    def bag_of_words(words):
      return dict([(word, True) for word in words.split()])

    #create bag of words (unordered) from newsarticle to be classified
    def bag_of_words2(words):
      #words = str(words)
      return dict([(word, True) for word in words])

    #create list of labeled feature set, with feature being a sentence from the corpus
    def label_feats_from_corpus(feature_detector=bag_of_words):
        sentence_polarity.ensure_loaded()
        label_feats = collections.defaultdict(list)
        for label in sentence_polarity.categories():
            t_para = sent_tokenize(sentence_polarity.raw(categories = label))
            for i in range(len(t_para)):
                feats = feature_detector(t_para[i])
                label_feats[label].append(feats)
        return label_feats

    #split labeled features
    def split_label_feats(lfeats, split=0.75):
      train_feats = []
      test_feats = []
      for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
      return train_feats, test_feats

    #randomly split labeled features
    def random_split_label_feats(lfeats, split=0.75):
      train_feats = []
      test_feats = []
      for label, feats in lfeats.items():
        train_feats.extend([(feat, label) for feat in feats])
      cutoff = int(len(train_feats) * split)
      random.shuffle(train_feats)
      test_feats = train_feats[cutoff:]
      train_feats = train_feats[:cutoff]
      return train_feats, test_feats
