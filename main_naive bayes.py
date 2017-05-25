from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from Polarity_NLTK.corpus_processing import Corpus
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.metrics import f_measure
import collections
from Polarity_NLTK.dbhandler2 import DatabaseHandler

#create corpus processor
cp = Corpus

#create sp_lfeats, sp_train_fests & sp_test_feats using sentence polarity corpus
sp_lfeats = cp.sp_label_feats_from_corpus()
sp_train_feats, sp_test_feats = cp.random_split_label_feats(sp_lfeats, split=0.75)
print(len(sp_train_feats))
print(len(sp_test_feats))
print(sp_train_feats)
print(sp_test_feats)

#create naivebayes classifier
sp_nb_classifier = NaiveBayesClassifier.train(sp_train_feats)


#get delta newsarticles from database, which did not get a sentiment yet, in
handler = DatabaseHandler()
result = handler.execute(
  """Select n.source_uri as 'source_uri', n.text as 'text' from NewsArticles n  WHERE n.source_uri NOT IN (Select s.source_uri FROM NewsArticlesNaiveBayes_SPSentiment s );
  """)

#write sentiments to db using training and testing set, persist dict
tuples = []
for row in result:
    i_text = row["text"]
    i_text = str(i_text)
    splitted_sentences = word_tokenize(i_text)
    splitted_sentences = cp.bag_of_words2(splitted_sentences)
    sent = sp_nb_classifier.classify(splitted_sentences)
    uri = row["source_uri"]
    uri = str(uri)
    processed ={}
    processed['source_uri'] = uri
    processed['sentiment'] = sent
    handler.persistDict('NewsArticlesNaiveBayes_SPSentiment', [processed])

#calculate accuracy, precision, recall and f measure for naive bayes classifier and most informative features
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(sp_test_feats):
    refsets[label].add(i)
    observed = sp_nb_classifier.classify(feats)
    testsets[observed].add(i)

pos_precision = precision(refsets['pos'], testsets['pos'])
pos_recall = recall(refsets['pos'], testsets['pos'])
pos_fmeasure = f_measure(refsets['pos'], testsets['pos'])
neg_precision = precision(refsets['neg'], testsets['neg'])
neg_recall = recall(refsets['neg'], testsets['neg'])
neg_fmeasure =  f_measure(refsets['neg'], testsets['neg'])

print(sp_nb_classifier.show_most_informative_features(15))

print('')
print('---------------------------------------')
print('RESULT ' + '(' + 'Naive Bayes Classifier on Sentence Polarity Dataset' + ')')
print('---------------------------------------')
print('accuracy:', accuracy(sp_nb_classifier, sp_test_feats))
print('precision', (pos_precision + neg_precision) / 2)
print('recall', (pos_recall + neg_recall) / 2)
print('f-measure', (pos_fmeasure + neg_fmeasure) / 2)
