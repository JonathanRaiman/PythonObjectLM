import gensim
from utils.covariance import objectlm_covariance
import gzip, pickle

def main():
    file = gzip.open("../saves/saved_texts.gz", 'r')
    texts, texts_data = pickle.load(file)
    file.close()

    doc2vec_corpus = [gensim.models.doc2vec.LabeledSentence(sentence, [str(k)]) for k,sentence in enumerate(texts)]
    doc2vec = gensim.models.Doc2Vec(doc2vec_corpus, dm=0, workers=4)

    for i in range(30):
        doc2vec.train(doc2vec_corpus)

    indices = [doc2vec.vocab[str(i)].index for i in range(len(texts))]
    objectlm_covariance(doc2vec.syn0norm[indices], "saves/pvdm")

if __name__ == '__main__':
    main()