import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from google_search import search
import warnings
warnings.filterwarnings("ignore")

# engine_id = "becd76ddad3b6ac04"
# API_KEY = "AIzaSyCVc0q0-3jtaEc__0I8GORSt2339A4mRsw"
query = "per se thomas keller"
# precision = 0.8

# Stem query so to get the vocabulary in query
def stem_query(query):
    query_vec = TfidfVectorizer()
    processed_query = query_vec.fit_transform([query])
    query_vocab = query_vec.get_feature_names_out()
    # print(query_vocab)
    return query_vocab

# Get stop words
def read_stop_words(query_vocab):
    stopwords = []
    with open('proj1-stop.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        # When adding stop word, we need to exclude those stop words that shows in query.
        for row in csv_reader:
            if row[0] not in query_vocab:
                stopwords.append(row[0])
            line_count += 1
    return stopwords

# calculating tf_idf and form vectors for document.
def cal_tf_idf(documents, query_vocab):
    doc_contents = []
    for d in documents:
        doc_contents.append(d['content'])
    stopwords = read_stop_words(query_vocab)
    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = vectorizer.fit_transform(doc_contents)
    collection_vocab = vectorizer.get_feature_names_out()
    return tfidf, collection_vocab
document = [{'content': 'Per Se | Thomas Keller Restaurant Group Per Se. Per Se Front Door. center. About. About; Restaurant · Team · Info & Directions · Gift Experiences · Reservations; Menus & Stories. Menus & Stories', 'label': 'y'}, {'content': 'Perse Definition & Meaning - Merriam-Webster We generally use per se to distinguish between something in its narrow sense and some larger thing that it represents. Thus, you may have no objection to\xa0...', 'label': 'y'}, {'content': 'per se - Wiktionary In and of itself; by itself; without determination by or involvement of extraneous factors; as such quotations ▽ · (chiefly in negative polarity environments)\xa0...', 'label': 'y'}, {'content': 'Per se - Wikipedia per se, a Latin phrase meaning "by itself" or "in itself". Illegal per se, the legal usage in criminal and antitrust law; Negligence per se, legal use in tort\xa0...', 'label': 'y'}, {'content': "Per Se - New York, NY | Tock Opened in 2004, Per Se is Thomas Keller's acclaimed interpretation of The French Laundry in the Deutsche Bank Center at Columbus Circle.", 'label': 'n'}, {'content': 'PER SE - 6798 Photos & 1719 Reviews - French - 10 Columbus Cir ... 1719 reviews of Per Se "This is Thomas Keller\'s new resturant in New York. A must if you can get a reservation."', 'label': 'n'}, {'content': 'Untitled 232k Followers, 190 Following, 738 Posts - See Instagram photos and videos from Per Se (@perseny)', 'label': 'n'}, {'content': 'per se | Wex | US Law | LII / Legal Information Institute For example, in tort law, a statutory violation is negligence per se. One of the elements a person has to prove in a negligence claim is that the defendant\xa0...', 'label': 'n'}, {'content': 'Per se - Definition, Meaning & Synonyms | Vocabulary.com Per se is the phrase to use when you want to refer to a particular thing on its own. It is not this Latin phrase, per se, that is important, but rather the\xa0...', 'label': 'y'}, {'content': 'PER SE | meaning in the Cambridge English Dictionary 3 days ago ... per se | American Dictionary ... by or of itself: It is not a pretty town per se, but it is where my family comes from, so I like it.', 'label': 'n'}]

if __name__ == '__main__':
  # doc, query = search()
  query_vocab = stem_query(query)
  tfidf, collection_vocab = cal_tf_idf(document, query_vocab)
  print(tfidf)
  print(collection_vocab)
  # print(tfidf.shape)
  print(len(collection_vocab))
# def stem_doc(doc):
#     splited_doc = []
#     words = word_tokenize(doc)
#     for w in words:
#         print(ps.stem(w))
#
# def count_vocab(documents):
#     colletion_vocab = defaultdict(int)
#
#     documents_vocab = []



