from nltk.corpus import sentence_polarity
from nltk.corpus import opinion_lexicon
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from new_start.dbhandler import DatabaseHandler
from new_start.Tokenizer import Tokenizer
from new_start.SP_Corpus_Processing import SP_Corpus
from nltk.classify import DecisionTreeClassifier
from nltk.probability import FreqDist, MLEProbDist, entropy
from new_start.OL_Corpus_Processing import OL_Corpus
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#create object of class SP_Corpus & processing the corpus
sp = SP_Corpus()
lfeats = SP_Corpus.label_feats_from_corpus()
print(lfeats.keys())
train_feats, test_feats = SP_Corpus.split_label_feats(lfeats, split=0.75)
print(train_feats)
print(test_feats)
#create naivebayes classifier and train classifier with training set
nb_classifier = NaiveBayesClassifier.train(train_feats)
#print(nb_classifier.labels())

#get newsarticles from database
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
result = handler.execute(
   """SELECT bow as 'text' FROM NewsArticlesBOW LIMIT 100
   """)

for i in (range(len(result))):
     i_text = result[i]
     i_text = str(i_text)
     splitted_sentences = word_tokenize(i_text)
     splitted_sentences = SP_Corpus.bag_of_words2(splitted_sentences)
     #print(splitted_sentences)
     print(nb_classifier.classify(splitted_sentences))


#tokenize newsarticel and classify it
#tokenizer = Tokenizer()
#for i in (range(len(result))):
#    i_text = result[i]
#    i_text = str(i_text)
#    splitted_sentences = tokenizer.split(i_text)
#    print(splitted_sentences)
    #splitted_sentences = SP_Corpus.bag_of_words2(splitted_sentences)
    #print(splitted_sentences)
    #print(nb_classifier.classify(splitted_sentences))


#measure performance of naive_bayes classifier
print(accuracy(nb_classifier, test_feats))
print(nb_classifier.show_most_informative_features(15))
