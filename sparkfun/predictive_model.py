import theano, theano.tensor as T, numpy as np
import theano_lstm
import pickle

def to_softmax(x):
    return T.nnet.softmax(x)[0] if x.ndim == 1 else T.nnet.softmax(x.T).T
class ProductModel(object):
    def __init__(self,
                 word_size,
                 vocabulary_size,
                 stack_size,
                 hidden_size,
                 hidden_price_size,
                 price_stack_size,
                 output_vocabulary,
                 memory_sparsity = 0.0001,
                 rho = 0.95,
                 verbose=False,
                 theano_mode = "FAST_RUN"):
        
        self.memory_sparsity= theano.shared(np.float64(memory_sparsity), name="memory_sparsity")
        self.theano_mode = theano_mode
        self.word_size = word_size
        self.vocabulary_size = theano.shared(np.int32(vocabulary_size), name="vocabulary_size")
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.output_vocabulary = output_vocabulary
        
        ### CREATE THE CELLS:
        
        model = theano_lstm.StackedCells(word_size, layers=[hidden_size] * stack_size, celltype=theano_lstm.LSTM, activation=T.tanh)
        # add a softmax layer at the end (non-recurrent)
        
        # special end token:
        model.layers.append(theano_lstm.Layer(hidden_size, output_vocabulary + 1, to_softmax))
        
        # add an embedding layer at the beginning (non-recurrent):
        model.layers = [theano_lstm.Embedding(vocabulary_size + output_vocabulary + 1, word_size),
                        theano_lstm.GatedInput(word_size, hidden_size, T.nnet.sigmoid)] + model.layers
        self.model = model
        
        model2 = theano_lstm.StackedCells(hidden_size, layers=[hidden_price_size] * (price_stack_size - 1) + [1], celltype=theano_lstm.Layer, activation=T.tanh)
        
        # price is a linear function of its inputs:
        model2.layers[-1].activation = T.exp
        
        self.price_model = model2
        
        
        ### CONSTRUCT THE PREDICTION / WIRING:
        
        def step(word_id, *prev_hiddens):
            if prev_hiddens[-1].ndim > 1:
                top_level_activ = prev_hiddens[-1][:, self.hidden_size:]
            else:
                top_level_activ = prev_hiddens[-1][self.hidden_size:]
            
            new_state = model.forward(word_id, [None, top_level_activ] + list(prev_hiddens), [])
            # all outputs should be returned, except embeddings, and the first gates
            return new_state[1:]
        
        def pred_step(word_id, *prev_hiddens):
            if prev_hiddens[-1].ndim > 1:
                top_level_activ = prev_hiddens[-1][:, self.hidden_size:]
            else:
                top_level_activ = prev_hiddens[-1][self.hidden_size:]
            new_state = model.forward(word_id, [None, top_level_activ] + list(prev_hiddens), [])
            # all outputs should be returned, except embeddings, and the first gates
            return [T.cast(new_state[-1].argmax() + self.vocabulary_size, dtype='int32')] + new_state[2:-1]
        
        def predict_sequence(x, lengths, return_all=False, return_memory=False):
            if x.ndim > 1:
                outputs_info = [None] + [dict(initial=T.repeat(T.shape_padleft(layer.initial_hidden_state), x.shape[0], axis=0), taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')]
            else:
                outputs_info = [None] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')]
            outputs_info = outputs_info + [None]
            result, updates = theano.scan(step,
                                          sequences = [x.T if x.ndim > 1 else x],
                                          outputs_info = outputs_info)

            if return_all:
                return result
            else:
                res = result[-1].dimshuffle(1, 0, 2) if x.ndim > 1 else result[-1]
                
                price_preds = self.price_model.forward(
                                self.model.layers[-2].postprocess_activation(
                                    result[-2][lengths, T.arange(0, lengths.shape[0])]
                                ), None, []
                            )[-1][:,0] if x.ndim > 1 else \
                            self.price_model.forward(
                                self.model.layers[-2].postprocess_activation(
                                    result[-2][-1]
                            ), None, [])[-1][0]
                # gate values can be obtained by asking for them from the stacked cells
                if return_memory:
                    return result[0], res, price_preds
                else:
                    return res, price_preds
        
        
        # every sequence is a series of indices
        # for words:
        input_sentences    = T.imatrix()
        
        # some sequences are shorter than others, so we'll note where they
        # end in a zero-indexed fashion
        sequence_lengths  = T.ivector()
        sequence_starts   = T.ivector()
        # the labels are integers in the range of dictionary
        
        self.input_sentences = input_sentences
        self.sequence_lengths = sequence_lengths
        self.sequence_starts = sequence_starts
        
        self.prices = T.vector()
        
        memory_usage, self.predictions, self.price_predictions = predict_sequence(input_sentences, self.sequence_starts, return_memory=True)
        
        self.error = (
            theano_lstm.masked_loss(
                self.predictions,
                input_sentences[:,1:] - self.vocabulary_size,
                sequence_lengths,
                sequence_starts).mean() +
            (memory_usage.sum() * self.memory_sparsity) / input_sentences.shape[0] +
            ((self.price_predictions - self.prices)**2).mean()
        )
        
        self.memory_fun = theano.function([input_sentences], memory_usage,
                                           allow_input_downcast=True,
                                           mode=self.theano_mode)
        
        self.price_predict_fun = theano.function([input_sentences, sequence_starts],
                                           self.price_predictions,
                                           allow_input_downcast=True,
                                           mode=self.theano_mode)
        
        self.predict_fun = theano.function([input_sentences],
                                           self.predictions,
                                           allow_input_downcast=True,
                                           mode=self.theano_mode)
        self.error_fun = theano.function([input_sentences, sequence_lengths, sequence_starts, self.prices],
                                         self.error,
                                         allow_input_downcast=True,
                                         mode=self.theano_mode)
        
        self.input_sentence = T.ivector()
        
        prep_result = predict_sequence(self.input_sentence, None, return_all=True)
        
        pred_outputs_info = [dict(initial=self.input_sentence[-1], taps=[-1])] + [dict(initial=prep_hidden[-1], taps=[-1]) for prep_hidden in prep_result[1:-1]]
        
        prediction_steps = T.iscalar()
        pred_result, _ = theano.scan(pred_step,
                                     n_steps = prediction_steps,
                                     outputs_info = pred_outputs_info)
        
        self.reconstruct_fun = theano.function([self.input_sentence, prediction_steps],
                                               pred_result[0],
                                               allow_input_downcast=True,
                                               mode=self.theano_mode)
        self.input_labels = theano.function([input_sentences],
                                            input_sentences[:,1:] - self.vocabulary_size,
                                            mode=self.theano_mode)
        
        if verbose:
            print("created prediction & error functions")
        updates, gsums, xsums, lr, max_norm = theano_lstm.create_optimization_updates(self.error, model.params + model2.params, max_norm=None, rho=rho, method="adadelta")
        self.lr = lr
        if verbose:
            print("took the gradient")
        
        self.gsums = gsums
        self.xsums = xsums
        
        self.update_fun = theano.function([input_sentences, sequence_lengths, sequence_starts, self.prices],
                                          outputs=None,
                                          updates=updates,
                                          mode=self.theano_mode)
        if verbose:
            print("created the gradient descent function")
    
    def reset_weights(self):
        for param, xsum, gsum in zip(self.model.params + self.model2.params, self.xsums, self.gsums):
            xsum.set_value(np.zeros_like(param.get_value(borrow=True)))
            gsum.set_value(np.zeros_like(param.get_value(borrow=True)))
            param.set_value((np.random.standard_normal(param.get_value(borrow=True).shape) * 1. /  param.get_value(borrow=True).shape[0]
                             ).astype(param.dtype))
            
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        f = open(path, "rb")
        self = pickle.load(f)
        f.close()
        return self