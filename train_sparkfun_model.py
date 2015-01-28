import gzip, pickle
import sparkfun
import argparse

def train_model(path, layers=4, epochs=100, hidden_size = 100, save_frequency=10):

    if path.find("{i}") == -1:
        raise ValueError("Output path should contain a format value for saving multiple iterations (e.g. \"saves/my_model_{i}/\"")

    if epochs < 1:
        raise ValueError("Can only train for 1 or more epochs.")

    if save_frequency >= epochs:
        save_frequency = epochs - 1

    with gzip.open("saves/sparkfun/products.pkl", "rb") as f:
        stuff = pickle.load(f)

    products = {}
    for product_args in stuff:
        product = sparkfun.Product(*product_args)
        products[product.sku] = product


    index2category, category2index, index2word, word2index = sparkfun.create_indices(products)
    data, codelens, sequence_lengths, prices = sparkfun.create_labeled_dataset(products, index2category, category2index, index2word, word2index, 10)

    # Construct a predictive model for pricing and categories
    # as a stacked LSTM system with an MLP on the last hidden
    # state outputting the price as a float:

    product_model = sparkfun.ProductModel(hidden_size,
        len(index2word), layers, hidden_size, hidden_size, 3, len(index2category),
                                 verbose=True,
                                 rho=0.95,
                                 memory_sparsity=0.3, theano_mode="FAST_COMPILE")

    def average_error():
        error = 0.
        size = 0
        for data_batch, sequence_lengths_batch, codelens_batch, prices_batch in zip(data, sequence_lengths, codelens, prices):
            error += product_model.error_fun(data_batch,
                                          codelens_batch,
                                     sequence_lengths_batch,
                                     prices_batch) * len(data_batch)
            size += len(data_batch)
        return error / size

    def training_loop():
        for data_batch, sequence_lengths_batch, codelens_batch, prices_batch in zip(data, sequence_lengths, codelens, prices):
            product_model.update_fun(data_batch,
                                    codelens_batch,
                                     sequence_lengths_batch,
                                     prices_batch)

    for epoch in range(epochs):
        training_loop()
        print("epoch %d : error %.3f" % (epoch, average_error()))
        if epoch > 0 and epoch % save_frequency == 0:
            product_model.save(path.format(i = epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'SparkFun Training',
        description='''
        Train StackedLSTM to predict price and categories for SparkFun products.
        ''')
    parser.add_argument(
        '-e', '--epochs', default = 5, metavar = 'Training Epochs', type=int,
        help = 'How many training epochs to train model for.')
    parser.add_argument(
        '-s', '--savefrequency', default = 10, metavar = 'Save Frequency', type=int,
        help = "How often to save the model's parameters")
    parser.add_argument(
        '-hid', '--hidden', default = 100, metavar = 'Hidden Size', type=int,
        help = "How big should the words, internal layers, and LSTM cells be ?")
    parser.add_argument(
        '-l', '--layers', default = 4, metavar = 'Layers', type=int,
        help = "How many Stacked LSTMs to use ?")
    parser.add_argument(
        '-o', '--output', metavar='Output path', type=str,
        help='Place to save model iteration saves', required = True)
    args = parser.parse_args()

    train_model(path=args.output, layers = args.layers, epochs=args.epochs, hidden_size=args.hidden,  save_frequency=args.savefrequency)