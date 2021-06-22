import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import io
import pandas as pd
from ctm import CollaborativeTopicModel
from sklearn.decomposition import LatentDirichletAllocation
from time import time
import gensim
import logging
from util import *
from os import path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    print('Running...')
    vocab_data = [line.rstrip('\n') for line in io.open('../data/citeulike/vocab.dat', encoding='utf8')]
    raw_data = pd.read_csv('../data/citeulike/raw-data.csv', sep=',', encoding="ISO-8859-1")
    
    if not path.exists("../data/citeulike/user-info.csv"):
        create_userinfo("../data/citeulike/users.dat", "../data/citeulike/user-info.csv")
    
    rating_data = pd.read_csv("../data/citeulike/user-info.csv")

    print('Data imported..')
    doc_ids, doc_word_ids, doc_word_cnts, ratings, raw_doc, tf_vec = raw_pd_to_cropus(vocab_data, raw_data, rating_data)

    # obtain beta,theta
    gensim_lda(tf_vec, raw_doc)
    print_doc_topics("../output/gensim.theta", exp=False)
    print('data prepared...')

    # train
    ctr = CollaborativeTopicModel(n_topic=200, n_voca=8000, doc_ids=doc_ids,
                                  doc_word_ids=doc_word_ids, doc_word_cnts=doc_word_cnts, ratings=ratings,
                                  beta_init='../output/gensim.beta', theta_init='../output/gensim.theta')
    ctr.fit(max_iter=50)

    # save ctr beta and theta
    ctr.save_theta_beta(beta_path='../output/ctr.beta',
                        theta_path='../output/ctr.theta', base_raw_doc_id=True)

    print_topics("../output/ctr.beta", '../data/citeulike/vocab.dat', exp=False)

    # new doc predict
    new_doc = "Researchers have access to large online archives of scientiﬁc articles. " \
              "As a consequence, ﬁnding relevant papers has become more difﬁcult. Newly formed online communities of researchers sharing citations provides a new way to solve this problem. In this paper, we develop an algorithm to recommend scientiﬁc articles to users of an online community. Our approach combines the merits of traditional collaborative ﬁltering and probabilistic topic modeling. It provides an interpretable latent structure for users and items, and can form recommendations about both existing and newly published articles. We study a large subset of data from CiteULike, a bibliography sharing service, and show that our algorithm provides a more effective recommender system than traditional collaborative ﬁltering."

    new_doc_ids, _ = raw_doc_to_cropus(tf_vec, new_doc)
    theta_init = doc_topic_distribution(new_doc, tf_vec)[0]
    r, theta = ctr.out_of_predict(0, new_doc_ids[0], theta_init=theta_init, return_theta=True)
    print_doc_topic(theta)
