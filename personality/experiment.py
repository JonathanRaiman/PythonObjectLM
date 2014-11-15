from .personality import Personality, CuisinePersonality
from .example_set import ExampleSet
import random

def create_personalities_examples(num_examples_per_personality = 12):
    """
    Experiment 1: create_personalities_examples
    -------------------------------------------

    Simulate diffirent selection behaviors over a corpus of personalities
    including `Cheap`, `Rating`, `Value`, and `Cuisine` where each 
    classifies as good or bad restaurants in the database on this criterium.

    The resulting classification is then used to generate example sets:
    sets of positive examples and sets of unknowns to classify.

    Surveyers can then use these positive examples to extrapolate and guess
    which one of the unknown examples is satisfactory.

    Note:
    -----
    A connection to MongoDB must be running for this method to work.

    Inputs
    ------

    num_examples_per_personality int: How many ExampleSets are generated for each of the
                                      personalities

    Outputs
    -------

    (personalities, example_sets) tuple: A list of the personalities used for generating examples,
                                         and a list of the generated examples.

    """
    cheap_personality = Personality(lambda el: len(el["price"]) < 2)

    # the `cuisine` behavior has these biases
    favorites = [
        (["burgers", "hotdogs"], []),
        (["vietnamese", "thai", "chinese", "korean", "japanese", "sushi", "szechuan", "taiwanese"],[]),
        (["italian", "mediterranean"],[]),
        (["mexican", "spanish", "argentine", "tex-mex"],[]),
        (["asianfusion", "french", "gluten_free", "gourmet", "asianfusion"], ["bakeries", "japanese", "pizza", "burgers", "sandwiches", "buffets"]),
        (["cafes","tea"], ["breakfast_brunch", "tradamerican"])
    ]

    cuisine_personalities = [CuisinePersonality(favorite) for favorite in favorites]

    rating_personality = Personality(lambda el: el["rating"] > 3.5)

    value_personality = Personality(lambda el: (el["rating"] - len(el["price"])) >= 1.9)
    personalities = []

    personalities.append(cheap_personality)
    personalities.extend(cuisine_personalities)
    personalities.append(rating_personality)
    personalities.append(value_personality)

    for personality in personalities:
        # each example will need 3 good examples, and then 1 good one along with 2 bad ones.
        personality.generate_samples(num_examples_per_personality * 4)
    
    example_sets = []
    for personality in personalities:
        pers_type = None
        if personality is cheap_personality:
            pers_type = 0
        if personality in cuisine_personalities:
            pers_type = 1
        if personality is rating_personality:
            pers_type = 2
        if personality is value_personality:
            pers_type = 3
        for i in range(num_examples_per_personality):
            options = [
                    personality.generate_bad_sample(),
                    personality.good_examples[i * 4 + 3],
                    personality.generate_bad_sample()
                ]
            random.shuffle(options)
            example_sets.append(ExampleSet(
                personality.good_examples[i*4: i*4 + 3],
                options,
                personality_type = pers_type,
                difficulty="easy"))
    return (personalities, example_sets)