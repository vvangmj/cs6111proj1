# COMS6111 Project 1

## Team Members
- Mingjun Wang (mw3542)
- Zihuan Wu (zw2771)

## Files
```
  -cs6111proj1
    - transcript.txt
    - main.py
    - count_vocab.py
    - google_search.py
    - proj1-stop.txt
    - rocchio.py
```
   
## How to Run
In order to run the system, you can run 
```
  python main.py --query "<query>" --precision <precision> [--engine_id "becd76ddad3b6ac04"] [--API_KEY "AIzaSyCVc0q0-3jtaEc__0I8GORSt2339A4mRsw"]
```
  or just
```
  python main.py --query "<query>" --precision <precision>
```

  - `<query>`: the search query string, which should be enclosed by quotes;
  - `<precision>`: the target for *precision@10*, a real number between 0 and 1;
  - `<engine_id>`: your engine ID. The default value is becd76ddad3b6ac04 (our engine id);
  - `<API_KEY>`: your API key. The default value is AIzaSyCVc0q0-3jtaEc__0I8GORSt2339A4mRsw (our API_KEY).
    
## Dependencies Installation
  - numpy: run ```pip install numpy==1.22.2```
  - sklearn: run ```pip install scikit-learn==1.0.2```
  - googleapiclient: run ```pip install google-api-python-client```
  - make sure your Python version is 3.7 or newer
  
## Internal Design
  Query modification works by using the vector space model to compare the relevant documents marked by the user, with the other documents returned by Google searched using the input query. This comparison is used to suggest new terms to add to the query, with the goal to obtain more relevant results. Vectors are computed according to each of the search results returned as well as each query generated. Note that when we take the exact top 10 results, we do ignore the non-html ones (i.e. pdf files) and our precision at each round will only focus on the html results. We ignore the non-html results by checking if the item has "fileFormat" keyword: if it does, then we will still present it but not count it into the final results.

## Query-modification method
  Our query modification works by first performing stemming on the input query and then calculating the TF-IDF scores based on the relevant and irrelevant results selected by users. After that, we run the Rocchio algorithm on the vector representing the input query along with parameter alpha = 1, beta = 0.8, gamma = 1 and TF-IDF results we obtained earlier to generate the vector for the next(extended) query. 
  
  To generate new query according to updated query vector, we get the top several words which have the greatest entries in query's vector, and sort those top words according to its weights(entries value) in descending order. And that will be the order of attempting adding the words into new query. To select at most two new word, we will not add those words already existed in the former query. Besides, after we have added one word to query, we will not take the second word into consideration and decide to only expand one word for final query if the second word has weights far less(difference>0.15) than the first added word. 
  
  Finally, after completing expansion, we return newly generated query to search and start the next round. 

## JSON API Key and Engine ID
- JSON API Key: ***hidden***
- Engine ID: ***hidden***
