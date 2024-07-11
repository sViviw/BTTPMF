import pandas as pd
from gensim.corpora.dictionary import Dictionary
import gzip
import re

def process_reviews(path, stopwords_path, sample_size=350, random_state=2):
    def parse(path):
        with gzip.open(path, 'rb') as g:
            for l in g:
                yield eval(l)

    def get_dataframe(path):
        df = {i: d for i, d in enumerate(parse(path))}
        return pd.DataFrame.from_dict(df, orient='index')

    stopwords_list = pd.read_csv(stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    stopwords_list = stopwords_list['stopword'].values.tolist()

    df = get_dataframe(path)

    df = df.loc[:, ['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['UserId', 'ItemId', 'Text', 'Rating']

    content = df['Text'].values.tolist()
    words = [[word for word in re.split('\W+', i) if word] for i in content]

    filtered_text_list = [[word for word in text if word.lower() not in stopwords_list] for text in words]

    df['Text'] = filtered_text_list

    df = df[[5 <= len(text) <= 20 for text in filtered_text_list]]

    df_sample = df.sample(n=sample_size, random_state=random_state)

    words_list = list(set(word for text in df_sample['Text'] for word in text))

    return df_sample, words_list

def get_dictionary(data):
    def preprocess_text(text):
        text = text.lower() 
        text = re.sub(r'\W+', ' ', text) 
        words = text.split()  
        return words

    # Preprocess texts
    docs = data["Text"]
    texts = [preprocess_text(doc) for doc in docs]

    # Create dictionary and corpus
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=2)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary,docs,texts



