import gensim
from gensim.corpora import TextCorpus
import gzip, pickle
from utils.covariance import objectlm_covariance

class TextCorpusSimple(TextCorpus):
    """
    Grab a list of pretokenized texts and create a corpus.
    """
    def get_texts(self):
        lineno = -1
        for lineno, line in enumerate(self.input):
            if self.metadata:
                yield line, (lineno,)
            else:
                yield line
        self.length = lineno + 1 # will be 0 if loop never executes

def main():
    file = gzip.open("../saves/saved_texts.gz", 'r')
    texts, texts_data = pickle.load(file)
    file.close()

    gensim_corpus = TextCorpusSimple(texts)
    tfidf = gensim.models.TfidfModel(gensim_corpus)
    lda = gensim.models.ldamodel.LdaModel(tfidf[gensim_corpus], num_topics=100)

    for topic in lda.show_topics(num_topics=30, formatted =False):
        for score, idea in topic:
            print("%.6f : %s" % (score, gensim_corpus.dictionary.get(int(idea))))
        print("\n===\n")

    y = lda.inference(tfidf[gensim_corpus])[0]
    objectlm_covariance(y, "saves/pLSA")

if __name__ == '__main__':
    main()