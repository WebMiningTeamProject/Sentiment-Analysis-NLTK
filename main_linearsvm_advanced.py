from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import sentence_polarity
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Polarity_NLTK.dbhandler import DatabaseHandler
import re
import nltk
from nltk.corpus import wordnet

###LinearSVM optimized for news articles (preprocessing of corpus by tokenizing, stop-word-removal and lemmatization & using and further optimizing tfidf-vectorizer and countvectorizer)

#preprocessing the corpus (tokenizing, stop-word-removal, lemmatization)
t_pos=[]
target = []
train = []
t_neg = []
tokenized_string = []
wordnet_lemmatizer = WordNetLemmatizer()
#preprocessing postive class movie reviews (POS Tags used for lemmatization)
t_pos = sent_tokenize(sentence_polarity.raw(categories = 'pos'))
for sent in t_pos:
    tokenized_word = word_tokenize(sent)
    tokenized_string = []
    pos_tagged_tokens = nltk.pos_tag(tokenized_word)
    for pos_tagged_token in pos_tagged_tokens:
        word, part_of_speech = pos_tagged_token
        if str(part_of_speech).startswith("N"):
            #token is a noun
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.NOUN))
        elif str(part_of_speech).startswith("V"):
            #token is a verb
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.VERB))
        elif str(part_of_speech).startswith("J"):
            #token is an adjective
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.ADJ))
        elif str(part_of_speech).startswith("R"):
            #token is adverb
            #note: lemmatizer does not handle adverbs
            extended_adverb = word + ".r.1"
            # handle the exception that no lemma is found
            try:
                lemmas = wordnet.synset(extended_adverb).lemmas()
                lemmatized_adverb = lemmas[0].pertainyms()[0].name()
            except (IndexError, AttributeError, nltk.corpus.reader.wordnet.WordNetError):
                lemmatized_adverb = word
            # add base form
            tokenized_string.append(lemmatized_adverb)
        else:
            #token is not tagged -> simply add token
            tokenized_string.append(word)

    # initialize return structure
    return_structure = []

    # delete everything except characters
    for token in tokenized_string:
        # replace with empty string
        return_structure.append(re.sub("[^a-züäößáàéè]", "", str(token).lower()))
        # remove empty entries
        return_structure = [x for x in return_structure if x]
    return_structure = str(return_structure)
    train.append(return_structure)
l = len(train)
#add positive target to target file
for i in range(len(train)):
    target.append('pos')

#preprocessing negative class movie reviews (POS Tags used for lemmatization)
t_neg = (sent_tokenize(sentence_polarity.raw(categories = 'neg')))
for sent in t_neg:
    tokenized_word = word_tokenize(sent)
    tokenized_string = []
    pos_tagged_tokens = nltk.pos_tag(tokenized_word)
    for pos_tagged_token in pos_tagged_tokens:
        word, part_of_speech = pos_tagged_token
        if str(part_of_speech).startswith("N"):
            #token is a noun
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.NOUN))
        elif str(part_of_speech).startswith("V"):
            #token is a verb
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.VERB))
        elif str(part_of_speech).startswith("J"):
            #token is an adjective
            tokenized_string.append(wordnet_lemmatizer.lemmatize(word, pos=wordnet.ADJ))
        elif str(part_of_speech).startswith("R"):
            #token is adverb
            #note: lemmatizer does not handle adverbs
            extended_adverb = word + ".r.1"
            # handle the exception that no lemma is found
            try:
                lemmas = wordnet.synset(extended_adverb).lemmas()
                lemmatized_adverb = lemmas[0].pertainyms()[0].name()
            except (IndexError, AttributeError, nltk.corpus.reader.wordnet.WordNetError):
                lemmatized_adverb = word
            # add base form
            tokenized_string.append(lemmatized_adverb)
        else:
            #token is not tagged -> simply add token
            tokenized_string.append(word)

    # initialize return structure
    return_structure = []

    # delete everything except characters
    for token in tokenized_string:
        # replace with empty string
        return_structure.append(re.sub("[^a-züäößáàéè]", "", str(token).lower()))
        # remove empty entries
        return_structure = [x for x in return_structure if x]
    return_structure = str(return_structure)
    train.append(return_structure)
#add negative target to target file
for i in range((len(train)) - l):
    target.append('neg')

#get logarithmic values for alpha to be tested during parameter tuning
alpha = []

for exponent in range(-15, 1):
    #add default and common value for alpha
    alpha.append(0.00001)
    alpha.append(0.000001)
    #add logarithmic numbers to alpha
    alpha.append(2 ** exponent)

#parameters to be tuned
params = {'vect__max_df': (0.5, 0.75, 1.0),
          'vect__max_features': (None, 5000, 10000, 50000),
          'vect__ngram_range': ((1, 1), (1, 2)),
          'tfidf__use_idf': (True, False),
          'tfidf__norm': ('l1', 'l2'),
          'clf__alpha': (0.0001, 0.0009765625),
          #'clf__alpha': (alpha),
          #'clf__penalty': ('l2', 'elasticnet'),
          #'clf__loss': ('hinge', 'squared_hinge'),
          #'clf__n_iter': (10, 50, 80)
          #'clf__fit_intercept': (True, False),
          #'clf__l1_ratio': (0.05, 0.1, 0.15, 0.2, 0.25),
          #'clf__power_t': (0.1, 0.25, 0.5, 0.75, 1)
           }

#pipeline to be processed by GridSearchCV
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

#parameter tuning using GridSearchCV
gridsearch = GridSearchCV(pipeline, params)
gridsearch.fit(train, target)
print(gridsearch.best_estimator_.get_params())
print(gridsearch.score(train,target))

## result
# {'steps': [('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=0.5, max_features=None, min_df=1,
#         ngram_range=(1, 2), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None)), ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)), ('clf', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
#        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
#        verbose=0, warm_start=False))], 'vect': CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=0.5, max_features=None, min_df=1,
#         ngram_range=(1, 2), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None), 'tfidf': TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True), 'clf': SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
#        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
#        verbose=0, warm_start=False), 'vect__analyzer': 'word', 'vect__binary': False, 'vect__decode_error': 'strict', 'vect__dtype': <class 'numpy.int64'>, 'vect__encoding': 'utf-8', 'vect__input': 'content', 'vect__lowercase': True, 'vect__max_df': 0.5, 'vect__max_features': None, 'vect__min_df': 1, 'vect__ngram_range': (1, 2), 'vect__preprocessor': None, 'vect__stop_words': None, 'vect__strip_accents': None, 'vect__token_pattern': '(?u)\\b\\w\\w+\\b', 'vect__tokenizer': None, 'vect__vocabulary': None, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': False, 'tfidf__use_idf': True, 'clf__alpha': 0.0001, 'clf__average': False, 'clf__class_weight': None, 'clf__epsilon': 0.1, 'clf__eta0': 0.0, 'clf__fit_intercept': True, 'clf__l1_ratio': 0.15, 'clf__learning_rate': 'optimal', 'clf__loss': 'hinge', 'clf__n_iter': 5, 'clf__n_jobs': 1, 'clf__penalty': 'l2', 'clf__power_t': 0.5, 'clf__random_state': None, 'clf__shuffle': True, 'clf__verbose': 0, 'clf__warm_start': False}
# 0.89364814041

#get delta newsarticles from database, which did not get a sentiment yet
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
result = handler.execute(
  """Select n.source_uri as 'source_uri', n.bow as 'bow' from NewsArticlesBOW n  WHERE n.source_uri NOT IN (Select s.source_uri FROM NewsArticlesLinearSVM_B s);
  """)

#add articles and predicted sentiments to database table and persist a dict
for row in result:
    i_text = (row['bow'])
    i_text = str(i_text)
    print(i_text)
    sent = gridsearch.predict([i_text])
    uri = row["source_uri"]
    uri = str(uri)
    if(sent == 'pos'):
        sent = 1
    elif(sent == 'neg'):
        sent = 0
    print(sent, uri)
    processed ={}
    processed['source_uri'] = uri
    processed['sentiment'] = sent
    handler.persistDict('NewsArticlesLinearSVM_B', [processed])

