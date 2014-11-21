import gzip, pickle, sys, argparse
sys.path.append("/Users/jonathanraiman/Desktop/Coding/language_modeling/")
from objectlm import ObjectLM, CategoriesConverter, DatasetGenerator
from word2vec_extended import Word2VecExtended


def train_model(path, bilinear = False, epochs = 5):

	if path.find("{i}") == -1:
		raise ValueError("Output path should contain a format value for saving multiple iterations (e.g. \"saves/my_model_{i}/\"")
	
	if epochs < 1:
		raise ValueError("Can only train for 1 or more epochs.")

	lmsenti = Word2VecExtended.load("/Users/jonathanraiman/Desktop/Coding/language_modeling/saves/kid_model_30_oov_senti")

	file = gzip.open("saves/saved_texts.gz", 'r')
	texts, texts_data = pickle.load(file)
	file.close()

	categories = set()
	for el in texts_data:
	    for c in el["categories"]:
	        categories.add(c)
	catconvert = CategoriesConverter(categories)
	dataset_gen = DatasetGenerator(texts, texts_data, catconvert)

	# should modify this to do some auto-encoding / self regression.
	model = ObjectLM(
	    vocabulary = lmsenti,
	    object_vocabulary_size = len(texts),
	    window = 10,
	    bilinear_form = bilinear,
	    size = 20,
	    object_size = 20,
	    output_sigmoid_classes = catconvert.num_categories,
	    output_sigmoid_labels = catconvert.index2category,
	    output_labels = [["", "$", "$$", "$$$", "$$$$"], ["1", "2", "3", "4", "5"]],
	    output_classes=[5, 5]
	)

	import logging
	logger = logging.getLogger("objectlm.training")
	logger.setLevel(logging.INFO)
	min_alpha = float(0.0001)
	max_alpha = float(0.0035)
	mega_errors = []

	for epoch in range(0, epochs):
	    alpha = max(min_alpha, max_alpha * (1. - (float(epoch) / float(epochs))))
	    model._alpha = alpha
	    objects, err = model.train(dataset_gen, workers = 8, chunksize = 24)
	    model.save_model(path.format(i = epoch))
	    print("Epoch = %d, Error = %.3f, alpha = %.3f" % (epoch, err, alpha))
	    mega_errors.append(err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'ObjectLM Training',
        description='''
        Train ObjectLM using text in bilinear or linear form with some savepath.
        ''')
    parser.add_argument('-b', '--bilinear', action = 'store_true',
        help = 'Use bilinear model.')
    parser.add_argument(
        '-e', '--epochs', default = 5, metavar = 'Training Epochs', type=int,
        help = 'How many training epochs to train model for.')
    parser.add_argument(
        '-o', '--output', metavar='Output path', type=str,
        help='Place to save model iteration saves', required = True)
    args = parser.parse_args()

    train_model(path=args.output, bilinear = args.bilinear, epochs=args.epochs)