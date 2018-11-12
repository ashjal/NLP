### 
# The main purpose of this snippet is to put unknown 
# documents into clusters.
# Creating clusters with each cluster having similar documents.
###

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import unicodedata

# path to all the files
path = ""
# list of document files
docs = []

# create a list of the contents of all the docs
dataArray = []
for doc in docs:
    newPath = path + doc
    raw = open(newPath, 'r')
    text = raw.read()
    data = unicodedata.normalize("NFKD", text)
    data = data.replace('\n', ' ')
    dataArray.append(data)

print(len(dataArray))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(dataArray)
print(X.todense())

# cluster size
true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

# Put your question below
Y = vectorizer.transform([""])
prediction = model.predict(Y)
print(prediction)