from googleapiclient.discovery import build

# engine_id = "becd76ddad3b6ac04"
# API_KEY = "AIzaSyCVc0q0-3jtaEc__0I8GORSt2339A4mRsw"
# query = "per se thomas keller"
precision = 0.8

def search(engine_id, API_KEY, query, precision):
    service = build("customsearch", "v1",developerKey=API_KEY)
    res = service.cse().list(q=query, cx=engine_id,).execute()

    print("Parameters:")
    print("Client key  = ", API_KEY)
    print("Engine key  = ", engine_id)
    print("Query       = ", query)
    print("Precision   = ", precision)
    print("Google Search Results:")
    print("======================")

    documents = []
    acc = 0
    counter = 1

    for item in res['items']:
        url = item['link']
        title = item['title']
        snippet = item['snippet']

        doc = {}

        print("Result ", counter)
        print("[")
        print("URL: ", url)
        print("Title: ", title)
        print("Summary: ", snippet)
        print("]\n")
        counter += 1

        doc['content'] = ' '.join([title, snippet])

        ans = input("Relevant (Y/N)? ")
        if ans.lower() == 'y':
            doc['label'] = 'y'
            acc += 1
            print()
        else:
            doc['label'] = 'n'
            print()
        documents.append(doc)

    cur_precision = acc/counter

    return query, documents, cur_precision

if __name__ == '__main__':
  doc = search()
  print(doc)
