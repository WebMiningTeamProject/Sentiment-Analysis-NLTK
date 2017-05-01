from nltk.tokenize import word_tokenize
import collections
from nltk.corpus import opinion_lexicon
import random

class OL_Corpus(object):
    #create a bag of words (unordered) from corpus
    def bag_of_words(words):
      return dict([(word, True) for word in words.split()])

    #create bag of words (unordered) from newsarticle to be classified
    def bag_of_words2(words):
      words = str(words)
      return dict([(word, True) for word in words.split()])

    def feats_from_corpus(feature_detector=bag_of_words):
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

    #split labeled features
    def split_label_feats(lfeats, split=0.75):
      train_feats = []
      test_feats = []
      for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
      return train_feats, test_feats

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
