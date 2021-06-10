import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.test.utils import datapath
import os
from sklearn.decomposition import LatentDirichletAllocation
import io
from time import time


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """
    pass


def safe_log(a):
    r = np.ma.log(a)
    return r.filled(-10000)


def raw_pd_to_cropus(vocab_data, raw_doc_data, rating_data):
    ratings = [(u, d) for u, d, _ in rating_data.itertuples(index=False)]
    raw_doc = [title + " " + abstract for _, title, _, _, abstract in raw_doc_data.itertuples(index=False)]
    tf_vec = CountVectorizer(vocabulary=vocab_data)
    tf_vec.fit(raw_doc)
    doc_ids = raw_doc_data['doc.id'].values
    doc_word_ids, doc_word_cnts = raw_doc_to_cropus(tf_vec, raw_doc)
    return doc_ids,doc_word_ids,doc_word_cnts, ratings, raw_doc, tf_vec




def raw_doc_to_cropus(tf, raw_doc):
    if isinstance(raw_doc, str): raw_doc = [raw_doc]
    tf_result = tf.transform(raw_doc)
    doc_word_ids = []
    doc_word_cnts = []
    for i in range(tf_result.shape[0]):
        ind_from = tf_result.indptr[i]
        ind_to = tf_result.indptr[i + 1]
        doc_word_ids.append(tf_result.indices[ind_from: ind_to])
        doc_word_cnts.append(tf_result.data[ind_from: ind_to])
    return doc_word_ids, doc_word_cnts


# for new document, use the previous model to find the probability which topic it belongs to
def doc_topic_distribution(new_doc, tf, ldamodel=None, lda_file="output/gensim/lda.model", inference=True, save_theta_path=None, save_beta_path=None):
    if isinstance(new_doc, str): new_doc = [new_doc]
    if lda_file is not None:
        temp_file = datapath(os.path.join(os.path.dirname(__file__), lda_file))
        ldamodel = gensim.models.ldamodel.LdaModel.load(temp_file)

    # Fit and transform
    X = tf.transform(new_doc)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    if inference:
        theta = ldamodel.inference(corpus)[0] # gamma distribution
    else:
        theta = list(ldamodel[corpus])  # only show some topics

    beta = ldamodel.get_topics()

    if save_theta_path:
        np.savetxt(save_theta_path, theta)
    if save_beta_path:
        np.savetxt(save_beta_path, beta)

    return theta, beta



def sklearn_lda(tf, raw_doc):
    tf_result = tf.transform(raw_doc)
    lda = LatentDirichletAllocation(n_components=200, max_iter=200,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0, batch_size=1000,
                                    verbose=True)
    t0 = time()
    W = lda.fit_transform(tf_result)  # theta doc-topic
    H = lda.components_  # beta topic-word
    print("done in %0.3fs." % (time() - t0))
    np.savetxt("../output/lda.theta", W)
    np.savetxt("../output/lda.beta", H)
    print_topics("../output/lda.beta", '../data/citeulike/vocab.dat', exp=False)


def gensim_lda(tf, raw_doc):
    tf_result = tf.transform(raw_doc)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(tf_result, documents_columns=False)
    id_map = dict((v, k) for k, v in tf.vocabulary_.items())

    # Use the gensim.models.ldamodel.LdaModel constructor to estimate
    # LDA model parameters on the corpus, and save to the variable `ldamodel`
    # Your code here:
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=200, id2word=id_map, chunksize=2000,
                                               passes=100, random_state=0)

    # Print Top 10 Topics / Word Distribution
    output = ldamodel.print_topics(20)
    print(output)

    np.savetxt("../output/gensim.beta", ldamodel.get_topics())
    np.savetxt("../output/gensim.theta", ldamodel.inference(corpus)[0])

    # lda = gensim.models.ldamodel.LdaModel.load(temp_file)
    ldamodel.save("../output/lda.model")


def print_topic(topic, vocabulary, exp=True, nwords=25):
    indices = list(range(len(vocabulary)))
    if exp:
        topic = np.exp(topic)
    topic = topic / topic.sum()
    indices.sort(key=lambda x: -topic[x])
    print(["{:}*{:.4f}".format(x, y) for (x, y) in zip(vocabulary[indices[0:nwords]], topic[indices[0:nwords]])])


def print_topics(beta_file, vocab_file, nwords=25, exp=True):
    # get the vocabulary
    vocabulary = np.array([line.rstrip('\n') for line in io.open(vocab_file, encoding='utf8')])
    topics = io.open(beta_file, 'r').readlines()
    # for each line in the beta file
    for topic_no,topic in enumerate(topics):
        print('topic %03d' % topic_no)
        topic = np.array(list(map(float, topic.split())))
        print_topic(topic, vocabulary, exp, nwords)
        print()


def print_doc_topic(doc_topic, exp=False, ntopic=20):
    indices = list(range(200))
    if exp: doc_topic = np.exp(doc_topic)
    doc_topic = doc_topic / doc_topic.sum()
    indices.sort(key=lambda x: -doc_topic[x])
    print(["Topic{:03d}*{:.4f}".format(x, y) for (x, y) in zip(indices[0:ntopic], doc_topic[indices[0:ntopic]])])


def print_doc_topics(theta_file, exp=False, ntopic=20):
    doc_topics = io.open(theta_file, 'r').readlines()
    # for each line in the beta file
    for doc_no, doc_topic in enumerate(doc_topics):
        print('doc %03d' % (doc_no+1))
        doc_topic = np.array(list(map(float, doc_topic.split())))
        print_doc_topic(doc_topic, exp=exp, ntopic=ntopic)
        print()
        if doc_no > 100: break