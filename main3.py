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

#tokenize and use dt_classifier
#dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.1, depth_cutoff=40, support_cutoff=1)
#for i in (range(len(result))):
#    i_text = result[i]
#    i_text = str(i_text)
#    print(i_text)
#    splitted_sentences = tokenizer.split(i_text)
#    splitted_sentences = C_NaiveBayes.bag_of_words2(splitted_sentences)
#    print(dt_classifier.classify(splitted_sentences))

#measure performance of decision tree classifier
#print(accuracy(dt_classifier, test_feats))
fd = FreqDist({'pos': 30, 'neg': 10})
#print(entropy(MLEProbDist(fd)))
fd['neg'] = 25
#print(entropy(MLEProbDist(fd)))
fd['neg'] = 30
#print(entropy(MLEProbDist(fd)))
fd['neg'] = 1
#print(entropy(MLEProbDist(fd)))

#try with dummi string
#negfeat = bag_of_words2(['alina', 'hates', 'the', 'bad', 'president'])
#print(nb_classifier.classify(negfeat))


#TODO: Accuracy und andere Measure berechnen und vergleichen
#TODO: Pruning of the words
