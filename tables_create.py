from new_start.dbhandler import DatabaseHandler
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
#result = handler.execute(
#     """CREATE TABLE NewsArticlesNaiveBayes_SPSentiment(source_uri varchar(511), sentiment varchar(6));
#   """)
#print(result)

# result = handler.execute(
#      """CREATE TABLE NewsArticlesNaiveBayes_OLSentiment (source_uri varchar(511) NOT NULL, sentiment varchar(6) NOT NULL, PRIMARY KEY (source_uri), CONSTRAINT fk_NewsArticlesNaiveBayes_OLSentiment_1 FOREIGN KEY (source_uri) REFERENCES NewsArticles (source_uri) ON DELETE NO ACTION ON UPDATE NO ACTION) ENGINE=InnoDB DEFAULT CHARSET=latin1;
#    """)
# print(result)

result = handler.execute(
     """CREATE TABLE NewsArticlesLinearSVM_I (source_uri varchar(511) NOT NULL, sentiment varchar(1) NOT NULL, PRIMARY KEY (source_uri), CONSTRAINT fk_NewsArticlesLinearSVM_1_I FOREIGN KEY (source_uri) REFERENCES NewsArticles (source_uri) ON DELETE NO ACTION ON UPDATE NO ACTION) ENGINE=InnoDB DEFAULT CHARSET=latin1;
   """)
print(result)


