MAP REDUCE

Dorjee Gyaltsen								                                                                                                    December 12, 2020


CSCI 496
Professor Lie Xie 

Summary

This program utilizes MapReduce design with a common web analysis algorithm, TF-IDF which is the Term frequency inverse document. Essentially it is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. In short we multiplied the term frequency of a specific term in that document with the inverse document frequency which is the logarithm of the n number of documents divided by the count of the term in each document. 


This results to the TF-IDF score which shows us the relevance of that term, allowing us to retrieve terms that are similar to the query term we have inputted. 

Sources and Requirements: 

- TF-IDF
For more info on TF-IDF, https://monkeylearn.com/blog/what-is-tf-idf/

- Apache Spark 
For this program we are using PyCharm, which we imported pyspark by downloading packages. 
If run through terminal download Apache Spark and follow the documentation, https://spark.apache.org/docs/latest/quick-start.html

- Python 3.9 (PyCharm)

Running the program

The programmed is run through PyCharm




