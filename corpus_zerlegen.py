from nltk.tokenize import sent_tokenize
from nltk.corpus import sentence_polarity
import collections
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from new_start.dbhandler import DatabaseHandler
from new_start.Tokenizer import Tokenizer

#create a bag of words (unordered) from corpus
def bag_of_words(words):
  return dict([(word, True) for word in words.split()])

#create bag of words (unordered) from newsarticle to be classified
def bag_of_words2(words):
  words = str(words)
  return dict([(word, True) for word in words])

#create list of labeled feature set, with feature being a sentence from the corpus
def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    corp.ensure_loaded()
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        t_para = sent_tokenize(corp.raw(categories = label))
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

#creating labeled feature set & check if labels are correct
lfeats = label_feats_from_corpus(sentence_polarity)
print(lfeats.keys())
#create train and test set & check if splitting worked out correctly
train_feats, test_feats = split_label_feats(lfeats, split=0.75)
print(len(train_feats))
print(len(test_feats))

#create naivebayes classifier and train classifier with training set
nb_classifier = NaiveBayesClassifier.train(train_feats)
print(nb_classifier.labels())

#get newsarticles from database
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
result = handler.execute(
   """SELECT bow FROM NewsArticlesBOW LIMIT 1
   """)
#tokenize newsarticel and classify it
result = str(result)
tokenizer = Tokenizer()
splitted_sentences = tokenizer.split(result)
splitted_sentences = bag_of_words2(splitted_sentences)
print(nb_classifier.classify(splitted_sentences))

#measure performance of classifier
print(accuracy(nb_classifier, test_feats))
print(nb_classifier.show_most_informative_features(15))



#try with dummi string
#negfeat = bag_of_words2(['alina', 'hates', 'the', 'bad', 'president'])
#print(nb_classifier.classify(negfeat))

