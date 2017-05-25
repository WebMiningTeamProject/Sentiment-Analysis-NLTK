from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import sentence_polarity
from nltk.tokenize import sent_tokenize
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from Polarity_NLTK.dbhandler2 import DatabaseHandler
from sklearn.feature_extraction.text import TfidfTransformer

###LinearSVM without additional preprocessing, with tfidf vectorizer and without tfidf vectorizer
#create list for training and list of target values
train=[]
target = []
#add positive sentences with polarity to training set
train = sent_tokenize(sentence_polarity.raw(categories = 'pos'))
l = len(train)
#add positive target to target file
for i in range(len(train)):
    target.append('pos')
#add negative sentences with polarity to training set
train.extend(sent_tokenize(sentence_polarity.raw(categories = 'neg')))
#add negative target to target file
for i in range(len(train)- l):
    target.append('neg')

#get logarithmic values for alpha to be tested during parameter tuning
alpha = []

for exponent in range(-15, 1):
    alpha.append(0.00001)
    alpha.append(0.000001)
    alpha.append(2 ** exponent)

#get parameters tuned for classifier only (approach referred to in the report)
# params = {'clf__alpha': (alpha),
#           'clf__penalty': ('l2', 'elasticnet'),
#           'clf__loss': ('hinge', 'squared_hinge'),
#           'clf__n_iter': (10, 50, 80),
#           'clf__fit_intercept': (True, False),
#           'clf__l1_ratio': (0.05, 0.1, 0.15, 0.2, 0.25),
#           'clf__power_t': (0.1, 0.25, 0.5, 0.75, 1)
#            }

#pipeline to be processed by GridSearchCV with classifer and countvectorizer only
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', SGDClassifier()),
# ])


#get parameters tuned for classifier, countvectorizer and tfidfvectorizer
params = {'vect__max_df': (0.5, 0.75, 1.0),
          'vect__max_features': (None, 5000, 10000, 50000),
          'vect__ngram_range': ((1, 1), (1, 2)),
          'tfidf__use_idf': (True, False),
          'tfidf__norm': ('l1', 'l2'),
          'clf__alpha': (alpha),
          'clf__penalty': ('l2', 'elasticnet'),
          'clf__loss': ('hinge', 'squared_hinge'),
          'clf__n_iter': (10, 50, 80),
          'clf__fit_intercept': (True, False),
          'clf__l1_ratio': (0.05, 0.1, 0.15, 0.2, 0.25),
          'clf__power_t': (0.1, 0.25, 0.5, 0.75, 1)
           }

#pipeline to be processed by GridSearchCV with classifer, tfidf vectorizer and countvectorizer only
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

#get delta newsarticles from database, which did not get a sentiment yet, in
handler = DatabaseHandler()
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
