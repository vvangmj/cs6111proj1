import numpy as np
import warnings
from scipy.sparse import csr_matrix
from count_vocab import *
from google_search import *

warnings.filterwarnings("ignore")

# document = [{'content': 'Per Se | Thomas Keller Restaurant Group Per Se. Per Se Front Door. center. About. About; Restaurant · Team · Info & Directions · Gift Experiences · Reservations; Menus & Stories. Menus & Stories', 'label': 'y'}, {'content': 'Perse Definition & Meaning - Merriam-Webster We generally use per se to distinguish between something in its narrow sense and some larger thing that it represents. Thus, you may have no objection to\xa0...', 'label': 'y'}, {'content': 'per se - Wiktionary In and of itself; by itself; without determination by or involvement of extraneous factors; as such quotations ▽ · (chiefly in negative polarity environments)\xa0...', 'label': 'y'}, {'content': 'Per se - Wikipedia per se, a Latin phrase meaning "by itself" or "in itself". Illegal per se, the legal usage in criminal and antitrust law; Negligence per se, legal use in tort\xa0...', 'label': 'y'}, {'content': "Per Se - New York, NY | Tock Opened in 2004, Per Se is Thomas Keller's acclaimed interpretation of The French Laundry in the Deutsche Bank Center at Columbus Circle.", 'label': 'n'}, {'content': 'PER SE - 6798 Photos & 1719 Reviews - French - 10 Columbus Cir ... 1719 reviews of Per Se "This is Thomas Keller\'s new resturant in New York. A must if you can get a reservation."', 'label': 'n'}, {'content': 'Untitled 232k Followers, 190 Following, 738 Posts - See Instagram photos and videos from Per Se (@perseny)', 'label': 'n'}, {'content': 'per se | Wex | US Law | LII / Legal Information Institute For example, in tort law, a statutory violation is negligence per se. One of the elements a person has to prove in a negligence claim is that the defendant\xa0...', 'label': 'n'}, {'content': 'Per se - Definition, Meaning & Synonyms | Vocabulary.com Per se is the phrase to use when you want to refer to a particular thing on its own. It is not this Latin phrase, per se, that is important, but rather the\xa0...', 'label': 'y'}, {'content': 'PER SE | meaning in the Cambridge English Dictionary 3 days ago ... per se | American Dictionary ... by or of itself: It is not a pretty town per se, but it is where my family comes from, so I like it.', 'label': 'n'}]


# resemble document labels
def document_label(document):
    label = []
    for doc in document:
        if(doc['label'] == 'y'):
            label.append(1)
        else:
            label.append(0)
    return label

# Initialize query vector
def initial_query(collection_vocab, query_vocab):
    V = len(collection_vocab)
    initial_query_vec = np.zeros(V)
    for idx in range(V):
        if collection_vocab[idx] in query_vocab:
            initial_query_vec[idx] = 1
    return initial_query_vec

# Implement rocchio's algorithm to update query vector
def rocchio(query_vec, doc_vec_csr, label):
    doc_vec = doc_vec_csr.toarray()

    V = len(doc_vec[0])
    sum_rel = np.zeros(V)
    sum_irrel = np.zeros(V)
    n_rel = 0
    n_irrel = 0

    for idx in range(len(label)):
        if label[idx] == 1:
            sum_rel += doc_vec[idx]
            n_rel += 1
        else:
            sum_irrel += doc_vec[idx]
            n_irrel += 1

    alpha = 1
    beta = 0.8
    gamma = 4

    rel_part = 0
    irrel_part = 0
    if n_rel != 0:
        rel_part = (beta/n_rel)*sum_rel
    if n_irrel!=0:
        irrel_part = (gamma/n_irrel)*sum_irrel

    query_vec = alpha*query_vec + rel_part - irrel_part

    return query_vec

# Generate new query according to updated query vector
def generate_query(query, query_vocab, query_vec, collection_vocab):
    query_size = len(query_vocab)
    # Get the top several words which have greatest weight in query's vector
    idx_top = np.argpartition(query_vec, -(query_size+2))[-(query_size+2):]
    count = 0
    for idx in idx_top:
        new_word = collection_vocab[idx]
        # print(query_vec[idx], new_word)
        if new_word not in query_vocab:
            query = ' '.join([query, new_word])
            count += 1
            if count == 2:
                break
    return query
# print(doc_vec[0].shape)
# print(initial_query.shape)
# print(initial_query)
# doc = [{'content': 'Per Se | Thomas Keller Restaurant Group Per Se. Per Se Front Door. center. About. About; Restaurant · Team · Info & Directions · Gift Experiences · Reservations; Menus & Stories. Menus & Stories', 'label': 'y'}, {'content': 'Perse Definition & Meaning - Merriam-Webster We generally use per se to distinguish between something in its narrow sense and some larger thing that it represents. Thus, you may have no objection to\xa0...', 'label': 'n'}, {'content': 'per se - Wiktionary In and of itself; by itself; without determination by or involvement of extraneous factors; as such quotations ▽ · (chiefly in negative polarity environments)\xa0...', 'label': 'n'}, {'content': 'Per se - Wikipedia per se, a Latin phrase meaning "by itself" or "in itself". Illegal per se, the legal usage in criminal and antitrust law; Negligence per se, legal use in tort\xa0...', 'label': 'n'}, {'content': "Per Se - New York, NY | Tock Opened in 2004, Per Se is Thomas Keller's acclaimed interpretation of The French Laundry in the Deutsche Bank Center at Columbus Circle.", 'label': 'y'}, {'content': 'PER SE - 6798 Photos & 1719 Reviews - French - 10 Columbus Cir ... 1719 reviews of Per Se "This is Thomas Keller\'s new resturant in New York. A must if you can get a reservation."', 'label': 'y'}, {'content': 'Untitled 232k Followers, 190 Following, 738 Posts - See Instagram photos and videos from Per Se (@perseny)', 'label': 'n'}, {'content': 'per se | Wex | US Law | LII / Legal Information Institute For example, in tort law, a statutory violation is negligence per se. One of the elements a person has to prove in a negligence claim is that the defendant\xa0...', 'label': 'n'}, {'content': 'Per se - Definition, Meaning & Synonyms | Vocabulary.com Per se is the phrase to use when you want to refer to a particular thing on its own. It is not this Latin phrase, per se, that is important, but rather the\xa0...', 'label': 'n'}, {'content': 'PER SE | meaning in the Cambridge English Dictionary 4 days ago ... per se | American Dictionary ... by or of itself: It is not a pretty town per se, but it is where my family comes from, so I like it.', 'label': 'n'}]
if __name__ == '__main__':
    query, doc, cur_pre = search()
    query_vocab = stem_query(query)
    doc_vec_csr, collection_vocab = cal_tf_idf(doc, query_vocab)
    label = document_label(doc)

    initial_query_vec = initial_query(collection_vocab, query_vocab)

    new_query_vec = rocchio(initial_query_vec, doc_vec_csr, label)

    query_str = generate_query(query, query_vocab, new_query_vec, collection_vocab)

    print(query_str)