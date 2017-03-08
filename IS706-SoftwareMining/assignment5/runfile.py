from sklearn.feature_extraction.text import TfidfVectorizer
import string
from os import listdir
from os.path import isfile, join
from scipy import spatial
import heapq


def preprocessing(pathFile):
    files = [f for f in listdir(pathFile) if isfile(join(pathFile, f))]
    lines = []
    for f in files:
        data = open(pathFile + '/' + f, 'r')
        string_ = [line.translate(None, string.punctuation).replace('\n', ' ').lower() for line in data]
        tokens = ''.join(string_).split()
        string_ = ' '.join(tokens)
        lines.append(string_.strip())
    return lines, files


def retrieveDocs(query, docs, nameDocs, nameQuery, numRetrieve):
    lengthDocs, ftrDocs = docs.shape
    results = [1 - spatial.distance.cosine(query.toarray(), docs[i].toarray()) for i in range(0, lengthDocs)]
    topRetrieve = heapq.nlargest(numRetrieve, zip(results, nameDocs))
    for top in topRetrieve:
        print nameQuery, top


if __name__ == '__main__':
    pathDocs = './documents'
    docs, nameDocs = preprocessing(pathDocs)
    pathQueries = './queries'
    queries, nameQueries = preprocessing(pathQueries)
    all = docs + queries
    vectorizer = TfidfVectorizer(min_df=20)
    X = vectorizer.fit_transform(all)
    docs, queries = X[:len(docs)], X[len(docs):]
    print X.shape, docs.shape, queries.shape
    for i in range(0, len(nameQueries)):
        retrieveDocs(queries[i], docs, nameDocs, nameQueries[i], 10)
