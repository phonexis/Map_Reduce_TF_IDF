"""
Dorjee Gyaltsen
Nusrat Chowdhury
Project 2

This program utilizes MapReduce design with a common web analysis algorithm,
TF-IDF - Term Frequency Inverse Document, is a statistical measure that evaluates
how relevant a word is to a document in a collection of documents. In short we multiply
the term frequency of the specific word to the inverse document frequency of that word
which will give us the TF-IDF score and will use to as a measure for similarity.
By using spark we can compute the TF-IDF.

"""
import math
import pyspark
import sys
import re

# Basic Load RDDs from external storage by calling a text-file
sc = pyspark.SparkContext('local[*]', 'tf_idf')

DIS_REGEX = re.compile('^(dis|gene)_[^ ]+_\\1$') # dis_breast_cancer_dis or gene_abc_gene
QUERY = ""

def txt_to_doc(txt):
    splitted = txt.split()
    # returns doc id and the word
    return splitted[0], [w for w in splitted[1:] if DIS_REGEX.match(w) or w == QUERY]

def doc_to_words(doc):
    words = doc[1]
    count_words = len(words)
    res = []
    for word in words:
        res.append(((doc[0], count_words, word), 1))
    return res

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Invalid try again.')
        exit(0)

    filename = sys.argv[1]

    QUERY = sys.argv[2]
    output = open('output', 'w')

    output.write(f'The term you entered is: {QUERY}\n')

    # inputting RDD through the function, with the result of the function
    # being the new value of each element in the resulting RDD.
    txt = sc.textFile(filename)
    docs = txt.map(txt_to_doc)

    doc_count = docs.count()
    #output.write(f'Document frequency: {doc_count}\n')
    # flatmap allows us to create a 1 to many relationships
    doc_words = docs.flatMap(doc_to_words) \
            .reduceByKey(lambda a, b: a + b) # term count per doc
    tf = doc_words.map(lambda word: (word[0][2], [(word[0][0], word[1]/word[0][1])])) \
            .reduceByKey(lambda a, b: a + b)

    # log function to find the idf and tf by dividing with the total length.
    tf_idf = tf.map(lambda word: (word[0],
        (math.log(doc_count / len(word[1]), 10), word[1])))

    # tf-idf computed by multiplying
    join_tf_idf = tf_idf.map(lambda word: (word[0], {i[0]: word[1][0] * i[1] for i in word[1][1]}))

    sort_tf_idf = join_tf_idf.sortByKey()
    q = sort_tf_idf.lookup(QUERY) # Return the list of values in the RDD for key. This operation is done efficiently if the RDD has a known partitioner by only searching the partition that the key maps to.

    q = [i for i in q]
    q_norm = sum(map(lambda x: x ** 2, q[0].values())) ** (1/2)

    similartity = join_tf_idf.map(lambda w: (w[0], sum([q[0][elem] * w[1][elem] for elem in q[0].keys() & w[1].keys()]) / (sum(map(lambda x: x ** 2, w[1].values())) ** (1/2) * q_norm)))

    # Get the N elements from an RDD ordered in descending order; Sorting by values
    terms = similartity.takeOrdered(11, key=lambda word: -word[1])

    output.write(f'\nTerm   tf-idf \n')

    output.writelines([f'{word} {item}\n' for (word, item) in terms if word != QUERY])
    output.close()
