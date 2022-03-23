# LDA Topic Modeling with Gibbs Sampling on Twitter Data

This project was created as part of the course **_'CS777 Big Data Analytics'_**.

Big Data analytics is the study of how to extract actionable, non-trivial knowledge from massive amount of data sets. The course focused both on the cluster computing software tools and programming techniques used by data scientists, as well as the important mathematical and statistical models that are used in learning from large-scale data processing.

I have used [PySpark](https://spark.apache.org/docs/latest/api/python/) to analyze huge amounts of data in my class assignments and project.

I ran the pyspark job on [Dataproc service](https://cloud.google.com/dataproc) by Google Cloud Platform which allows the capability to interactively analyze data in a distributed environment.

>_Note: Technically I didn't need to use pyspark interface to analyze the twitter data as the number of tweets being analyzed are small (~ 31 K), but this serves as a learning experince on how to analyze large data in a distributed environment._

## Goal
Discover hidden semantics, themes or abstract "topics" in twitter data related to covid-19

## Latent Dirichlet Allocation (LDA)
One of the most popular topic modeling algorithms

- **Latent**:  This refers to everything that we don’t know a priori and are hidden in the data. Here, the themes or topics that document consists of are unknown.
- **Dirichlet**: In the context of topic modeling, the Dirichlet is the distribution of topics in documents and distribution of words in the topic.
- **Allocation**: This means that once we have Dirichlet, we will allocate topics to the documents and words of the document to topics.

Let’s fix some parameters:

- T = number of topics
- D = number of documents
- V = different words in the vocabulary
- N = number of words in each doc, so doc i has Ni words
- W<sub>ij</sub> = specific j-th word in document i
- z<sub>ij</sub> = topic of j-th word in doc i

Let θ be the distribution of topics over the documents, then θ<sub>i</sub> is the topic distribution for document I.

Let Ф be the distribution of words over topics, then Ф<sub>k</sub> is the word distribution for topic k.

We will model the distribution of topics over the documents θ and the word distribution for topics Ф by Dirichlet distributions of order T and V respectively.

Other than that, we have α, the parameter of the Dirichlet prior on the per-document topic distributions and β, the parameter of the Dirichlet prior on the per-topic word distribution.

### **So how do we find θ and Ф ?? :eyes: By [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) !!**

To start with, we will assume that we know ϴ and Ф matrices. We will then slowly change these matrices and get to an answer that maximizes the likelihood of the data that we have. We will do this on word-by-word basis by changing the topic assignment of one word. We will assume that we don’t know the topic assignment of the given word, but we know the assignment of all other words in the text, and we will try to infer what topic will be assigned to this word.

Mathematically, we do the sampling of a new topic z<sub>ij</sub> for a word w<sub>ij</sub> by:
![image](https://user-images.githubusercontent.com/64467745/157758925-f2e769b2-ba7b-452d-966e-7c06be3f2b39.png)

We will do this on word-by-word basis by changing the topic assignment of one word until the cover the whole corpus. This would be one iteration. We can iterate over multiple times to improve the likelihood.

While reasearching for LDA, I came across this amazing video on YouTube which explains LDA very clearly, check it out if you want!

[![LDA](https://img.youtube.com/vi/T05t-SqKArY/hqdefault.jpg)](http://www.youtube.com/watch?v=T05t-SqKArY)


## Twitter Data
- I have generated Twitter data using the Twitter API. You would require a Twitter API token to request the data. Learn how to generate the token and use API [here](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api)
- Here I have captured tweets between couple of days
- My twitter API account only allows max 100 tweets to be returned in one request, hence I have sent multiple requests to cover the days
- I have gathered data that contain **_"covid-19"_** text
- After cleaning the data (removing emoji’s, URL, usernames etc.), I have about 31K tweets

Python script that I wrote to get the Twitter data can be found [here](src/ExtractTwitterData.py)

Raw Twitter data obtained via API can be found [here](data/data.csv)

Cleaned data can be found [here](data/final_twitter_data.csv)

> _Note: Please create your own Twitter API token and use it in the script_

## LDA with Gibbs Sampling
- First converted the corpus into dictionary of top _n_ words, here I have used _n_ = 10000
- Created a RDD which contains
  - Array of positions of words in dictionary for each doc
  - Array of initial topics assigned to each word for each doc
  - Array of number of topics assigned to each word for each doc (this will become our Ѳ distribution; distribution of topics over documents)
  - An integer denoting number of words in each doc
- Created a RDD which contains frequency of words for each topic (this will be used for creating the Ф distribution; word distribution for topics)

Python code for LDA with Gibbs Sampling can be found [here](src/LDATopicModelingWithGibbs.py)

## Result
After Gibbs sampling, we can see the topic distribution among the docs. For e.g., see below example for few documents

![image](https://user-images.githubusercontent.com/64467745/157761947-475c4819-47f1-48e5-8c0a-e84046c4edf8.png)

Here we observe that for:
- Document 1, there is 60% prob that it belongs to topic 3, 20% for topic 1 and 2
- Document 6, more than 80% prob belonging to topic 1 and about 20% of topic 5
- Similarly, we can observe findings for other documents as well

Below are the top 20 words in each of the topics (I gave an input of clustering words into 5 topics)

![image](https://user-images.githubusercontent.com/64467745/157762552-7b808910-8a29-4495-bc2d-ca96b623b363.png)

- Topic1: This topic looks to be about children and school and how covid is impacting the education
- Topic2: This looks to be about workers/police officers impacted
- Topic3: This looks about covid impact versus India
- Topic4: This looks to be about vaccine shots, vaccine companies, booster shots
- Topic5: This looks to be about the studies/research about covid-19

## Limitation
During the calculation of gibbs sampling, the pyspark code was becoming very complex. It is because, we need to sample each word in document and update the θ and φ distributions, which were part of rdd. Since rdd itself is immutable, and cannot be iterated over sequentially, I had to create number of rdd’s and it was becoming too complex. Due to time constraint, I was not able to research more on this and thus, I’m doing all the Gibbs sampling calculation in pure python code.

## Conclusion
This small project just scratches the surface of huge topic of LDA and Gibbs Sampling. The results that are obtained can be improved further by say setting more optimal number of topics which can be determined by computing Coherence Score or Hierarchical Dirichlet Process (HDP). These are complex methods out of scope of this project for now.

## Word to the reader
Please feel to review the material and use it for your own further research. I would appreciate any feedback on this project.
