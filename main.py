from nltk.corpus import sentence_polarity
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from new_start.dbhandler import DatabaseHandler
from new_start.Tokenizer import Tokenizer
from new_start.C_NaiveBayes import C_NaiveBayes
from nltk.classify import DecisionTreeClassifier

#create object of class naivebayes classifer
classifier = C_NaiveBayes()
#creating labeled feature set & check if labels are correct
lfeats = C_NaiveBayes.label_feats_from_corpus(sentence_polarity)
print(lfeats.keys())
#create train and test set & check if splitting worked out correctly
train_feats, test_feats = C_NaiveBayes.split_label_feats(lfeats, split=0.75)
print(len(train_feats))
print(len(test_feats))

#create naivebayes classifier and train classifier with training set
nb_classifier = NaiveBayesClassifier.train(train_feats)
print(nb_classifier.labels())

#get newsarticles from database
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
result = handler.execute(
   """SELECT bow as 'text' FROM NewsArticlesBOW LIMIT 3
   """)

#tokenize newsarticel and classify it
tokenizer = Tokenizer()
for i in (range(len(result))):
    i_text = result[i]
    i_text = str(i_text)
    print(i_text)
    splitted_sentences = tokenizer.split(i_text)
    splitted_sentences = C_NaiveBayes.bag_of_words2(splitted_sentences)
    print(nb_classifier.classify(splitted_sentences))

#measure performance of naive_bayes classifier
print(accuracy(nb_classifier, test_feats))
print(nb_classifier.show_most_informative_features(15))

#tokenize and use dt_classifier
dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
for i in (range(len(result))):
    i_text = result[i]
    i_text = str(i_text)
    print(i_text)
    splitted_sentences = tokenizer.split(i_text)
    splitted_sentences = C_NaiveBayes.bag_of_words2(splitted_sentences)
    print(dt_classifier.classify(splitted_sentences))

#measure performance of decision tree classifier
print(accuracy(dt_classifier, test_feats))


#try with dummi string
#negfeat = bag_of_words2(['alina', 'hates', 'the', 'bad', 'president'])
#print(nb_classifier.classify(negfeat))

