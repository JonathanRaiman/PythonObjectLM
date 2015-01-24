from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import gzip, pickle
import numpy as np
from objectlm import CategoriesConverter
from utils.covariance import objectlm_covariance

def get_catconverter(texts_data):
    categories = set()
    for el in texts_data:
        for c in el["categories"]:
            categories.add(c)
    catconvert = CategoriesConverter(categories)
    return catconvert

def tfidf_process_text(texts):
    vectorizer = TfidfVectorizer(input = str,
                                 strip_accents = 'ascii',
                                 analyzer ='word',
                                 max_features=5000)
    y = vectorizer.fit_transform(" ".join(text) for text in texts)
    return y

def get_labels(texts_data):
    price_data = np.array([len(text["price"]) for text in texts_data])
    rating_data = np.array([np.round(text["rating"]) for text in texts_data])

    catconvert = get_catconverter(texts_data)
    category_data = np.array([catconvert.category2index[text["categories"][0]] for text in texts_data])
    return price_data, rating_data, category_data

def main():
    file = gzip.open("../saves/saved_texts.gz", 'r')
    texts, texts_data = pickle.load(file)
    file.close()
    price_data, rating_data, category_data = get_labels(texts_data)
    y = tfidf_process_text(texts)

    svm_price = LinearSVC(C=4, dual=False)
    svm_price.fit(y, price_data)
    y_trans_price = svm_price.transform(y, threshold = "2*mean").toarray()
    svm_rating = LinearSVC(C=4, dual=False)
    svm_rating.fit(y, rating_data)
    y_trans_rating = svm_rating.transform(y, threshold = "2*mean").toarray()
    svm_category = LinearSVC(C=4, dual=False)
    svm_category.fit(y, category_data)
    y_trans_category = svm_category.transform(y, threshold = "2*mean").toarray()
    y_trans = np.hstack([y_trans_category, y_trans_price, y_trans_rating])
    objectlm_covariance(y_trans + 1e-12, "../saves/svm", metric="euclidean")

if __name__ == '__main__':
    main()