import gzip, pickle
import sparkfun
import argparse
import sys
sys.setrecursionlimit(50000)
import theano
import time

theano.config.reoptimize_unpickled_function = False
theano.gof.compilelock.set_lock_status(False)

def test_model(path):
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

    product_model = sparkfun.ProductModel.load(path)

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
    t0 = time.time()
    err = average_error()
    t1 = time.time()
    print("error %.3f, evaluation time %.3fs" % (err, t1 - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'SparkFun Training',
        description='''
        Train StackedLSTM to predict price and categories for SparkFun products.
        ''')
    parser.add_argument(
        '-m', '--model', metavar='Model path', type=str,
        help='Where to load the model from', required = True)
    args = parser.parse_args()

    test_model(args.model)