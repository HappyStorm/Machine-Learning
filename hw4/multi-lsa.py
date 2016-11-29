# coding: utf-8
import csv, sys, time
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

n_svd = 20

def load_data(dirpath):
    # list of strings
    with open(dirpath + 'title_StackOverflow.txt') as file:
    # with open('../data/title_StackOverflow.txt') as file:
        corpus = [s.strip().lower() for s in file.readlines()]

    # list of triples (index, id_1, id_2)
    with open(dirpath + 'check_index.csv') as file:
    # with open('../data/check_index.csv') as file:
        test_data = file.readlines()[1:]
        test_data = [map(int, s.strip().split(',')) for s in test_data]
    return corpus, test_data

start = time.time() # Start time

print 'Loading data...'
corpus, test_data = load_data(sys.argv[1])
print 'Data loading... ---Done---'

tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
tfidf_vectorized = tfidfvectorizer.fit_transform(corpus)
count_vectorized = countvectorizer.fit_transform(corpus)
multi_features = sp.sparse.hstack([tfidf_vectorized, count_vectorized]).toarray()

svd = TruncatedSVD(n_svd)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

lsa_multi_features = lsa.fit_transform(multi_features)

print 'Start K-Means clustering...'
kmeans = KMeans(n_clusters=100, init='k-means++', max_iter=1000, n_init=100).fit(lsa_multi_features)
labels = kmeans.labels_
print 'K-Means clustering... ---Done---'

print 'Writing answer...'
with open(sys.argv[2], 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(('ID', 'Ans'))
    for i, s in enumerate(test_data):
        ans = 1 if labels[s[1]] == labels[s[2]] else 0
        writer.writerow((i, ans))
print 'Answer writing... ---Done---'

end = time.time()
elapsed = end - start
print "Time taken for program: ", elapsed, "seconds."