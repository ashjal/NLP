# NLP
Small NLP POCs using python.

1. #DocumentClustering
Given a number of documents on varying unknown topics, cluster them into given number of clusters. Following clustering, basis the question raised, point to the cluster that should containn the answer.

2. #SingleFileQ&A
To build a Question & Answering system (using Google's Universal Sentence Encoder, the brain behind TalkToBooks). Currently putting all the text data in a single file.
Flow:
Read text -> Convert each sentence to embedding (vector) -> Convert question to embedding -> Calculate similarity of each document embedding with question embedding -> Print most similar result.

