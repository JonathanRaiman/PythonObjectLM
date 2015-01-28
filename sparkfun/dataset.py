import numpy as np
from xml_cleaner import to_raw_text

def sparsity(x):
    if type(x) is list:
        total_nonzero = 0.
        total_size = 0.
        for batch in x:
            total_nonzero += len(batch.nonzero()[0])
            total_size += batch.shape[0] * batch.shape[1]

        return total_nonzero / total_size
    else:
        return len(x.nonzero()[0]) / (x.shape[0] * x.shape[1])

def create_indices(data, mincount=5):
    index2category = []
    category2index = {}
    word2index = {}
    index2word = []
    wordcounts = {}

    for product in data.values():
        for category in product.categories:
            if category not in category2index:
                category2index[category] = len(index2category)
                index2category.append(category)
        
        if not (type(product.description) == list):
            des = []
            for line in to_raw_text(product.description):
                for word in line:
                    des.append(word)
                des.append("\n")
            product.description = des
        tokens = product.description

        for word in tokens:
            if word not in wordcounts:
                wordcounts[word] = 1
            else:
                wordcounts[word] += 1

                
    for word, occurence in wordcounts.items():
        if occurence >= mincount:
            word2index[word] = len(index2word)
            index2word.append(word)
    
    word2index["**UNKNOWN**"] = len(index2word)
    index2word.append("**UNKNOWN**")
    
    word2index["**END**"] = len(index2word)
    index2word.append("**END**")
            
    return index2category, category2index, index2word, word2index

def create_labeled_dataset(products, index2category, category2index, index2word, word2index, subpieces = 2):
    dataset = []
    prices = np.empty(len(products), dtype='float32')
    codelens = np.empty(len(products), dtype='int32')
    sequence_lengths = np.empty(len(products), dtype='int32')
    vocab_size = len(index2word)
    i = 0
    for product in products.values():
        prices[i] = product.price
        
        # create the sequence target:
        index_line = []
        for word in product.description:
            if word in word2index:
                index_line.append(word2index[word])
            else:
                index_line.append(word2index["**UNKNOWN**"])
        index_line.append(word2index["**END**"])
        for category in product.categories:
            index_line.append(vocab_size + category2index[category])
        index_line.append(word2index["**END**"])
        dataset.append(index_line)
        
        codelens[i] = len(product.categories) + 1
        sequence_lengths[i] = len(product.description)
        
        i+=1
        
    
    lengths = np.array(list(map(len, dataset)))
    
    shortest = np.argsort(lengths)
    
    piece_size = np.ceil(len(lengths) / subpieces)
    
    so_far = 0
    
    data_batches = []
    codelens_batches = []
    sequence_lengths_batches = []
    prices_batches = []
    
    dataset = np.array(dataset, dtype=object)
    
    for i in range(subpieces):
        indices = shortest[so_far : so_far + piece_size]
        
        max_len_example = lengths[indices].max()
        data = np.zeros([len(indices), max_len_example], dtype=np.int32)
        
        for k, example in enumerate(dataset[indices]):
            data[k,:len(example)] = example
            
            
        codelens_batches.append(codelens[indices])
        sequence_lengths_batches.append(sequence_lengths[indices])
        prices_batches.append(prices[indices])
        
        so_far += piece_size
        data_batches.append(data)

        
    return data_batches, codelens_batches, sequence_lengths_batches, prices_batches