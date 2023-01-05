from count_vocab import *
from google_search import *
from rocchio import *

engine_id = "becd76ddad3b6ac04"
API_KEY = "AIzaSyCVc0q0-3jtaEc__0I8GORSt2339A4mRsw"
query = ""
precision = 0.9

if __name__ == '__main__':

    query, doc, cur_precision = search(engine_id, API_KEY, query, precision)
    while(cur_precision < precision):
        query_vocab = stem_query(query)
        doc_vec_csr, collection_vocab = cal_tf_idf(doc, query_vocab)
        label = document_label(doc)

        initial_query_vec = initial_query(collection_vocab, query_vocab)

        new_query_vec = rocchio(initial_query_vec, doc_vec_csr, label)

        query_str = generate_query(query, query_vocab, new_query_vec, collection_vocab)

        print("New Query:", query_str)
        query, doc, cur_precision = search(engine_id, API_KEY, query_str, precision)

    # TODO:FEEDBACK SUMMARY
    print("Desired precision reached, done!")