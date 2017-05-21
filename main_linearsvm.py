from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import sentence_polarity
from nltk.tokenize import sent_tokenize
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from Polarity_NLTK.dbhandler2 import DatabaseHandler

t_para=[]
t_para2 = []
#for label in sentence_polarity.categories():
t_para = sent_tokenize(sentence_polarity.raw(categories = 'pos'))
l = len(t_para)
print(l)
for i in range(len(t_para)):
    t_para2.append('pos')
t_para.extend(sent_tokenize(sentence_polarity.raw(categories = 'neg')))
print(len(t_para))
for i in range(len(t_para)- l):
    t_para2.append('neg')
print(t_para)
print(t_para2)
print(len(t_para))
print(len(t_para2))

parameters = {'C': (1, 10)}
svm_classifier = SklearnClassifier(svm.NuSVC())


# params = {'clf__alpha': (0.00001, 0.000001, 0.5, 1.0),
#           'clf__penalty': ('l2', 'elasticnet'),
#           'clf__loss': ('hinge', 'squared_hinge')}
alpha = []

for exponent in range(-15, 1):
    alpha.append(0.00001)
    alpha.append(0.000001)
    alpha.append(2 ** exponent)
print(alpha)
params = {'clf__alpha': (alpha),
          'clf__penalty': ('l2', 'elasticnet'),
          'clf__loss': ('hinge', 'squared_hinge'),
          'clf__fit_intercept': (True, False),
          'clf__l1_ratio': (0.05, 0.1, 0.15, 0.2, 0.25),
          'clf__power_t': (0.1, 0.25, 0.5, 0.75, 1)}

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SGDClassifier()),
])

# svr = svm.SVC()
gridsearch = GridSearchCV(pipeline, params)
gridsearch.fit(t_para, t_para2)
print(gridsearch.best_estimator_.get_params())
print(gridsearch.score(t_para,t_para2))

# handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
# result1 = handler.execute(
#   """SELECT source_uri as 'source_uri', bow as 'bow' FROM NewsArticlesBOW LIMIT 10
#   """)

 #get delta newsarticles from database, which did not get a sentiment yet, in
handler = DatabaseHandler()
result = handler.execute(
  """Select n.source_uri as 'source_uri', n.bow as 'bow' from NewsArticlesBOW n  WHERE n.source_uri NOT IN (Select s.source_uri FROM NewsArticlesLinearSVM_I s);
  """)


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
    handler.persistDict('NewsArticlesLinearSVM_I', [processed])
