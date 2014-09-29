Yelp Language Model
===================

A language model for documents and words. This simple language model takes a supervised input dataset with pairs of labels in multinomial or unimodal classes, and then trains the documents and windows of words within those documents to predict those labels. Both documents and words are trained through backprop. Words are shared among all documents, while document vectors live in their own embedding.

Eucledian distances between documents are observed to possess fuzzy search properties over all the labels.

## Custom Language Model

Here we initialize a model that uses the vectors for the words in a window and a special object vector corresponding to the document (restaurant) to perform classification. By gradient descent we can then update the word vectors and the object vectors so that the object vectors obtain some relation to the labels / targets provided to us (in this case the Yelp category, pricing, and rating labels).
    
    model = YelpLM(
        vocabulary = lmsenti,
        object_vocabulary_size = len(texts),
        window = 10,
        bilinear_form = False,
        size = 20,
        object_size = 20,
        output_sigmoid_classes = catconvert.num_categories,
        output_sigmoid_labels = catconvert.index2category,
        output_classes=[5, 5], # "", "$", "$$",...,"$$$$", 5 price classes, and 5 rating classes
        output_labels = [["", "$", "$$", "$$$", "$$$$"], ["1", "2", "3", "4", "5"]]
    )

    import logging
    logger = logging.getLogger("yelplm.training")
    logger.setLevel(logging.INFO - 1)
    #observation_work = np.zeros(model.window * model.size + model.object_size, dtype = np.float32)
    #distribution_work = np.zeros(model.output_size, dtype = np.float32)
    min_alpha = float(0.001)
    max_alpha = float(0.0035)
    max_epoch = 9
    for epoch in range(0, max_epoch):
        alpha = max(min_alpha, max_alpha * (1. - (float(epoch) / float(max_epoch))))
        model._alpha = alpha
        objects, err = model.train(dataset_gen, workers = 8, chunksize = 24)
        #for example in dataset_gen:
        #    total_error += train_sentence_concatenation(model, np.array([model.vocab.get_index(word) for word in example[0]], dtype=np.int32), example[1], example[2], example[3], alpha, distribution_work, observation_work)
        print("Error = %.3f, alpha = %.3f" % (err, alpha))


We can then perform gradient descent on all the examples and minimize the classification error for each object. Running this for about 9 epochs works for a small dataset, and hopefully applies to the larger case here.

In this particular instance we find that looking at the eucledian distance between object vectors acts as a fuzzy search on all the attributes. It remains to be evaluated how much of the semantic information about the objects is contained in these. Furthermore, this model is not auto-regressive, thus there is no way to generalize to unlabeled data in the future. Nonetheless for document retrieval purposes this is effective.

It is important to note that there are hundreds of labels to predict, but only 20 dimensions for the object vector, thus this enforces specificity.

A Java implementation can be [found here](https://github.com/JonathanRaiman/objectlm).

### Get a dataset of restaurants

Either get the data from mongo:


    from utils import connect_to_database, get_some_restaurants
    connect_to_database(database_name = 'yelp')
    max_el = 10000
    texts, texts_data = get_some_restaurants(max_el, min_words = 10) # be as large as a single window.
    with gzip.open("saves/saved_texts.gz", 'wb') as file:
        pickle.dump((texts, texts_data), file, 1)

or load it from the drive:


    file = gzip.open("saves/saved_texts.gz", 'r')
    texts, texts_data = pickle.load(file)
    file.close()

The model can then be built by discovering which labels are needed (pricing,
rating, and categories in this case), and from this data we build the `ObjectLM`
model:


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
        bilinear_form = False,
        size = 20,
        object_size = 20,
        output_sigmoid_classes = catconvert.num_categories,
        output_sigmoid_labels = catconvert.index2category,
        output_labels = [["", "$", "$$", "$$$", "$$$$"], ["1", "2", "3", "4", "5"]],
        output_classes=[5, 5] # "", "$", "$$",...,"$$$$", 5 price classes, and 5 rating classes
    )

### Training the model

Training is then a matter of going through the dataset in any order and lowering
the training rate as we aim to reduce the prediction error over the labels for
each restaurant while taking as input the restaurant vectors and the text
windows.

    import logging
    logger = logging.getLogger("objectlm.training")
    logger.setLevel(logging.INFO)
    min_alpha = float(0.0001)
    max_alpha = float(0.0035)
    max_epoch = 9
    for epoch in range(0, max_epoch):
        alpha = max(min_alpha, max_alpha * (1. - (float(epoch) / float(max_epoch))))
        model._alpha = alpha
        objects, err = model.train(dataset_gen, workers = 8, chunksize = 24)
        model.save_model("saves/objectlm_window_10_lm_20_objlm_20_%d/" % (epoch))
        print("Epoch = %d, Error = %.3f, alpha = %.3f" % (epoch, err, alpha))

After 4 epochs training diverges catastrophically, so stop it beforehand. Overfitting is an issue, but there is no real way to generalize this data to anything, so this is not something we need to cross-validate or somehow prevent urgently.


### Saving the model or exporting to Matlab / Java:

The model's parameters can be saved to interact with Java as follows:

    model.save_model_to_java("saves/current_model")

Then from Java you can import this model as described [here](https://git.mers.csail.mit.edu/jraiman/yelplm/tree/master#load-language-model).


#### Additional Exports

Other files can be saved separately for exporting purposes

    dataset_gen.save("saves/objectlm_window_10_lm_20_objlm_20_4/__dataset__.gz")
    dataset_gen.save_ids("saves/objectlm_window_10_lm_20_objlm_20_4/__objects__.gz")
    model.save_model_parameters("saves/objectlm_window_10_lm_20_objlm_20_4")
    model.save_model_to_java("saves/objectlm_window_10_lm_20_objlm_20_4")
    catconvert.save_to_java("saves/objectlm_window_10_lm_20_objlm_20_4/__categories__.gz")
    model.save_vocabulary("saves/objectlm_window_10_lm_20_objlm_20_4")


### Loading a saved model:


To load a saved model, point it to a directory with the saved matrices:

    model.load_saved_weights("saves/objectlm_window_10_lm_20_objlm_20_4/")


### Querying the model


First we create normalized matrices:

    model.create_normalized_matrices()

Then we can perform searches on it using inner product distance (cosine):

    model.most_similar_word("science")

    [('request', 0.9545077085494995, 7163),
     ('hopefully', 0.9531156420707703, 38713),
     ('community', 0.9526830911636353, 6000),
     ('infused', 0.9511095285415649, 7513),
     ('yummy', 0.9509859085083008, 34636),
     ('fallen', 0.9509795904159546, 6096),
     ('feeling', 0.9508317708969116, 38029),
     ("'d", 0.9483151435852051, 26839),
     ('reading', 0.9478667974472046, 20754),
     ('work', 0.9475015997886658, 586)]


That's not brilliant, however the objects have captured better properties:


    def most_similar_object(self, vector, topn = 20):
            dists = np.dot(self.norm_object_matrix, vector).astype(np.float32)
            best = np.argsort(dists)[::-1][:topn]
            result = [(sim, float(dists[sim])) for sim in best]
            return result[:topn]
    def search_for_object_with_vector(self, vector, topn = 20):
        display(HTML("<h2>Search with Vector</h2>"))
        for result, distance in most_similar_object(self, vector, topn=topn):
            present_restaurant(texts_data[result], text = texts[result])
            display(HTML("<small>%d</small>" % result))
            display(HTML("""
                <div>
                    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
                        <div style='height:20px; width:%dpx;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
                            %.0f%%
                        </div>
                    </div>
                </div>
                """ % (int(distance * 100), distance * 100)))


    from utils import present_restaurant
    from IPython.display import display, HTML
    import re
    
    def levenshtein_distance(source, target):
        if len(source) < len(target):
            return levenshtein_distance(target, source)
     
        # So now we have len(source) >= len(target).
        if len(target) == 0:
            return len(source)
     
        # We call tuple() to force strings to be used as sequences
        # ('c', 'a', 't', 's') - numpy uses them as values by default.
        source = np.array(tuple(source))
        target = np.array(tuple(target))
     
        # We use a dynamic programming algorithm, but with the
        # added optimization that we only need the last two rows
        # of the matrix.
        previous_row = np.arange(target.size + 1)
        for s in source:
            # Insertion (target grows longer than source):
            current_row = previous_row + 1
     
            # Substitution or matching:
            # Target and source items are aligned, and either
            # are different (cost of 1), or are the same (cost of 0).
            current_row[1:] = np.minimum(
                    current_row[1:],
                    np.add(previous_row[:-1], target != s))
     
            # Deletion (target grows shorter than source):
            current_row[1:] = np.minimum(
                    current_row[1:],
                    current_row[0:-1] + 1)
     
            previous_row = current_row
     
        return previous_row[-1]
    
    def search_for_object(self, object_index, topn = 10):
        present_restaurant(texts_data[object_index], text = texts[object_index])
        display(HTML("<small>%d</small>" % object_index))
        for result, distance in self.most_similar_object(object_index, topn=topn):
            present_restaurant(texts_data[result], text = texts[result])
            display(HTML("<small>%d</small>" % result))
            display(HTML("""
                <div>
                    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
                        <div style='height:20px; width:%dpx;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
                            %.0f%%
                        </div>
                    </div>
                </div>
                """ % (int(distance * 100), distance * 100)))
    def search_with_text(self, text, topn = 10, levenshtein = False):
        text = text.lower()
        min_distance_index = -1
        min_distance = float('inf')
        min_distance_word = None
        
        for i, datum in enumerate(texts_data):
            if levenshtein:
                min_local_distance = float('inf')
                if datum["_id"].lower().find(text) != -1:
                    min_local_distance = 3
                    if min_local_distance < min_distance:
                        min_distance_word = datum["_id"]
                        min_distance = min_local_distance
                        min_distance_index = i
                for scrap in re.split( "[ -]", datum["_id"].lower()) + [datum["_id"].lower()] + datum["categories"]:
                    min_local_distance = min(min_local_distance, levenshtein_distance(text, scrap))
                    if min_local_distance < min_distance:
                        min_distance_word = scrap
                        min_distance = min_local_distance
                        min_distance_index = i
                if min_distance <= 1:
                    break
            else:
                if datum["_id"].lower().find(text) != -1 or datum["id"].lower().find(text) != -1:
                    min_distance = i
                    break
                for cat in datum["categories"]:
                    if cat.lower().find(text) != -1:
                        min_distance = i
                        break
                
        if min_distance_index == -1 or min_distance == len(text) or min_distance == len(min_distance_word):
            print("Could not be found")
        else:
            if min_distance > 0:
                display(HTML("""
                <span style="color: #333">Did you mean </span> <b>%s</b> <span style="color: #333">(%d edit%s)</span> ?
                """ % (min_distance_word, min_distance, "s" if min_distance != 1 else "")))
            return search_for_object(self, min_distance_index, topn=topn)


Testing the levenshtein distance for our search:

    search_with_text(model, "Tangerine Thai", levenshtein= True, topn=5)


And indeed we get some matching places:

<div>
    <h2>Tangerine Thai</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>6673 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>review ... ? seattle , wa this is a cute little spot on phinney , with a great wall that completely opens up on nice days . the food is</p>
</div>
<small>1796</small>
<div>
    <h2>Mae Ploy Thai Cuisine</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>5157 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>the fried tofu was perfect as well . '' '' the curry was spicy and full of vegetables . '' '' the staff is always very nice and recognize me</p>
</div>
<small>1911</small>
<div>
    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
        <div style='height:20px; width:98px;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
            99%
        </div>
    </div>
</div>
<div>
    <h2>Pailin Thai Cuisine</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>5445 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>k . '' love the fish tank , really cool to look at . '' '' i 'm kind of in a rut , since i usually get the phad</p>
</div>
<small>2055</small>
<div>
    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
        <div style='height:20px; width:98px;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
            98%
        </div>
    </div>
</div>
<div>
    <h2>Thailanding On Alki</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>7453 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>of all the service was amazing ! ! ! they genuinely cared if we liked the food or not . second , the food was great . i had the</p>
</div>
<small>2030</small>
<div>
    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
        <div style='height:20px; width:98px;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
            98%
        </div>
    </div>
</div>
<div>
    <h2>Ying Thai Kitchen</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>7349 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>. they are very happy about this place 's food which means good sign ! was this review ... ? seattle , wa i had not had thai food in</p>
</div>
<small>1928</small>
<div>
    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
        <div style='height:20px; width:97px;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
            98%
        </div>
    </div>
</div>
<div>
    <h2>Royal Palm Thai Restaurant</h2>
    <span style='color: #ca0814;'>$$</span><span style='color: #e4e4e4;'>$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>5487 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>thai</span>
    <p style='width:450px'>driving as i wanted to try a different place . i ordered the 4 star out of 5 , gluten-free , gang ped ( red curry , duck , pineapple</p>
</div>
<small>1949</small>
<div>
    <div style='background: rgb(221,222,223);width:102px;padding:1px;border-radius:2px'>
        <div style='height:20px; width:97px;padding-left:3px;background-color:rgb(71, 189, 249);font-size:9px;color:white'>
            98%
        </div>
    </div>
</div>
        
With some spelling mistakes:


    search_with_text(model, "burgez", levenshtein= True, topn=5)


We still get a correction and a result:

<span style="color: #333">Did you mean </span> <b>burger</b> <span style="color: #333">(1 edit)</span> ?
            
<div>
    <h2>burger-king-seattle-6</h2>
    <span style='color: #ca0814;'>$</span><span style='color: #e4e4e4;'>$$</span><br /> 
    <span style='color: #feea60;'>★★★</span><span style='color: #e4e4e4;'>★★</span>
    <span style='color:#333;font-size:9px'>1059 words</span>
    <br/>
    <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>burgers, hotdogs</span>
    <p style='width:450px'>lunch . i met with two other people here during the rush and i 'm surprised that all of us made it out alive as first-timers . i had a</p>
</div>


### Dependencies

You will probably want the [xml_cleaner](https://github.com/JonathanRaiman/xml_cleaner) for cleaning up text if you want to easily process weirdly formatted inputs.

