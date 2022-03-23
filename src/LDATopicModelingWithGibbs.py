import sys
import numpy as np
import string
from operator import add
from pyspark import SparkContext
from sklearn.feature_extraction import text
from collections import Counter

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: LDAGibbsSampling <DataFile> <Vocab_size> <Number_of_Topics>', file=sys.stderr)
        exit(-1)
    try:
        int(sys.argv[2])
    except ValueError:
        print('Usage: Vocab_size must be integer')
        exit(-1)
    try:
        int(sys.argv[3])
    except ValueError:
        print('Usage: Number_of_Topics must be integer')
        exit(-1)

# Create the spark context
sc = SparkContext()

# Read data file into RDDs
rdd = sc.textFile(sys.argv[1])

# Map RDD to (docid, text)
rdd = rdd.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))

# Map RDD to (docid, [word1, word2 ...])
rdd = rdd.map(lambda x: (x[0], list(x[1].split(' '))))

# Stopwords that we will remove from the text
custom_stopwords = ['', 'im', 'just', 'like', 'want', 'people', 'say', 'says', 'tell', 'tells', 'whats']
stopwords = text.ENGLISH_STOP_WORDS.union(custom_stopwords).union(list(string.ascii_lowercase))

# Removing stopwords
rdd = rdd.map(lambda x: (x[0], [word for word in x[1] if word not in stopwords]))

# Create RDD [(word1, 1), (word2, 1) ... ]
allWords = rdd.flatMap(lambda x: x[1]).map(lambda x: (x, 1))

# This will give us the count of each word in the corpus
# [(word1, 123), (word2, 321) ...]
allCounts = allWords.reduceByKey(add)

# vocab size
V = int(sys.argv[2])

# Get the top V frequency words (sorted) in the corpus
topWords = allCounts.top(V, lambda x: x[1])

# Print top 10 words in the corpus by frequency
print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through V
# V is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(V))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID = rdd.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID) pairs
allDictionaryWords = allWordsWithDocID.join(dictionary)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x: (x[1][0], x[1][1]))


def my_add(x, y):
    x.append(y)
    return x


# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.aggregateByKey(list(), my_add, lambda x, y: x + y)

# Get a RDD ([dicPos1, dicPos2, ....], [dicPos1, dicPos2, ...], ...)
allDictionaryWordsInEachDoc = allDictionaryWordsInEachDoc.map(lambda x: x[1])

# Number of documents
D = allDictionaryWordsInEachDoc.count()
# Number of topics
T = int(sys.argv[3])

# the parameter of the Dirichlet prior on the per-document topic distributions
alpha = 1 / T
# the parameter of the Dirichlet prior on the per-topic word distribution
beta = 1 / T

# For Gibbs Sampling, we do the sampling of a new topic 'z' (topic for the j-th word in document i)
# for a word 'w' (specific j-th word in i-th document) using the formula:
# P(z[ij] | z[kl] with k!=i and l!=j,w) = ((theeta + alpha)/(N[i] + alpha*T))((phi + beta)/(Sigma(phi) + beta*V))
# Here theeta = topic distribution for i-th document
# phi = word distribution for topic k
# N[i] = Number of words in i-th document


# First map assigns a topic randomly to word, here we are taking the modulo of index of word / Number_of_topic
# to assign one topic between 0 and T-1
# Second map will create an array which defines the number of words associated with each topic for that document
# and also add total number of words in the document
# Below will be the rdd structure for one document
# [(
# [word1PosInDic, word2PosInDic, ...],
# [word11Topic, word2Topic, ...],
# [numberOfWordsAssignedTopic0, numberOfWordsAssignedTopic1, ...]
# [totalWordsInDocument]
# ), ...]
finalrdd = allDictionaryWordsInEachDoc.map(lambda x: (x, [k % T for k in range(len(x))])). \
    map(lambda x: (x[0], x[1], [Counter(x[1])[k] for k in range(T)], len(x[0])))

# Cache final rdd
finalrdd.cache()
finalrdd.count()

# Calculate the initial word distribution for topics based on final rdd
# 1st map creates RDD = [(topicForWord1inDoc1, correspondingPosInDic), (topicForWord2inDoc1, correspondingPosInDic),...
# (topicForWord1inDoc2, correspondingPosInDic), (topicForWord2inDoc2, correspondingPosInDic), ...
# (topicForWord1inDocD, correspondingPosInDic), (topicForWord2inDocD, correspondingPosInDic)]
# 2nd map creates RDD where the x[1] position is filled with V zeros (except where 'correspondingPosInDic' matches
# the index, it places 1)
# [(topicForWord1inDoc1, [0,0,0,... 1(at position index=correspondingPosInDic) ...0,0,0])]

phi_z_w_rdd = finalrdd.flatMap(lambda x: ((x[1][d], x[0][d]) for d in range(len(x[0])))). \
    map(lambda x: (x[0], [1 if x[1] == d else 0 for d in range(V)]))


def add_list(x, y):
    return np.add(x, y)


# Reduce the rdd to get the word distribution for topics
# [(0, [frequencyOfWordsAtDicPos0ForTopic0, frequencyOfWordsAtDicPos1ForTopic0, ...),
# (1, [frequencyOfWordsAtDicPos0ForTopic1, frequencyOfWordsAtDicPos1ForTopic1,...),
# ...
# (T-1, [frequencyOfWordsAtDicPos0ForTopic(T-1), frequencyOfWordsAtDicPos1ForTopic(T-1)...)]
phi_z_w_rdd_reduce = phi_z_w_rdd.reduceByKey(add_list)

# Note that the pyspark code for Gibbs sampling was becoming too complicated
# and because of time limitation as well, I'll be using normal python code for now
wordPos = finalrdd.map(lambda x: x[0]).collect()  # word position for each doc
z_dn = finalrdd.map(lambda x: x[1]).collect()  # topics assigned to each word in a doc
theta_dz = np.array(finalrdd.map(lambda x: x[2]).collect())  # number of words assigned to each topic (topic distribution by document)
phi_zw = np.array(phi_z_w_rdd_reduce.map(lambda x: x[1]).collect())   # word distribution by topic
n_d = np.array(finalrdd.map(lambda x: x[3]).collect())  # total words in each document
n_z = np.array(finalrdd.map(lambda x: x[2]).reduce(add_list))   # total word topics

# Number of iterations
n_iterations = 10

for iteration in range(n_iterations):
    for i, doc in enumerate(wordPos):
        for j, w in enumerate(doc):
            # get the topic for word n in document d
            z = z_dn[i][j]

            # decrement counts for word w with associated topic z
            theta_dz[i][z] -= 1
            phi_zw[z, w] -= 1
            n_z[z] -= 1

            # sample new topic from a multinomial according to formula
            p_d_t = (theta_dz[i] + alpha) / (n_d[i] - 1 + T * alpha)
            p_t_w = (phi_zw[:, w] + beta) / (n_z + V * beta)
            p_z = p_d_t * p_t_w
            p_z /= np.sum(p_z)
            # Sample a new topic for the word 'j' in document 'i'
            new_z = np.random.multinomial(1, p_z).argmax()

            # set z as the new topic and increment counts
            z_dn[i][j] = new_z
            theta_dz[i][new_z] += 1
            phi_zw[new_z, w] += 1
            n_z[new_z] += 1

    print(f'Gibbs Sampling in progress... Iteration {iteration+1} completed!')

print('Gibbs sampling completed...')
# Collect the dictionary
vocabulary = dictionary.collect()
# Inverse of vocabulary
inv_vocabulary = {v: k for k, v in dict(vocabulary).items()}

# Top n words for each topic
n_top_words = 20
m = []
print(f'Top {n_top_words} words in each of the {T} topics:')
for topic_idx, topic in enumerate(phi_zw):
    message = f'Topic {topic_idx+1}: '
    message += ' '.join([inv_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)

sc.stop()
