import random 
from utils import connect_to_database
from pymongo.errors import ConnectionFailure
from warnings import warn

try:
    DB = connect_to_database(database_name = "yelp")
except ConnectionFailure as e:
    warn("Could not connect to MongoDB database `yelp`")
    DB = None

class Personality:
    def __init__(self, good_requirement, collection_name = "restaurants"):
        self.collection_name = collection_name
        self.good_requirement = good_requirement
        self.good_examples = []
        
    def _random_sample(self):
        database = DB[self.collection_name]
        all_elements = database.count()
        random_el = random.randint(0, all_elements)
        els = list(database.find({"review_count": {"$gt": 4}}, {"url": 1,"rating":1, "price": 1, "categories":1}, limit=1, skip=random_el))
        if len(els) > 0:
            return els[0]
        else:
            return self._random_sample()
        
    def generate_bad_sample(self):
        sample = self._random_sample() 
        if self.good_requirement(sample):
            return self.generate_bad_sample()
        else:
            return sample
        
    def generate_sample(self):
        sample = self._random_sample() 
        if self.good_requirement(sample):
            return sample
        else:
            return self.generate_sample()
            
    def generate_samples(self, count=100):
        self.good_examples = [self.generate_sample() for i in range(count)]
            
class CuisinePersonality(Personality):
    def __init__(self, favorites, collection_name = "restaurants"):
        likes, dislikes = favorites
        self.likes = set(likes)
        self.dislikes = set(dislikes)
        self.collection_name = collection_name
        self.good_examples = []
        
    def good_requirement(self, example):
        if len(set(example["categories"]) & self.likes) > 0 and len(set(example["categories"]) & self.dislikes) == 0:
            return True
        return False