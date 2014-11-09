from IPython.display import display, HTML
import re
from xml_cleaner import to_raw_text
from pymongo import MongoClient
from math import ceil
import numpy as np

mongo_database_global = None
mongo_client_global   = None

def get_mongo_client(hostname = 'localhost', port = 27017):
    global mongo_client_global
    mongo_client_global = MongoClient(hostname, port)

def connect_to_database(database_name='test'):
    global mongo_client_global
    global mongo_database_global
    if not mongo_client_global:
        get_mongo_client()
    mongo_database_global = mongo_client_global[database_name]

class NotConnectedError(Exception):
    def __init__(self, message):
        self.message = message

def restaurant_saved_text_preprocess(text):
    """
    Isolate sentences into words and groups them into a single stream.
    """
    
    text = text.replace("Copyright © 2004-2014 Yelp Inc. Yelp, , and related marks are registered trademarks of Yelp.", "").replace("Was this review ...?", "").replace("This user has arrived from Qype, the newest addition to the Yelp family. The Yelp & Qype engineering team is hard at work integrating the two sites, so stay tuned! Thanks for your patience.", "")
    text = re.sub("([.,!?;])([0-9a-zA-Z])", "\g<1> \g<2>", re.sub("([0-9a-zA-Z])([.,!?;])", "\g<1> \g<2>", text))
    
    words = [word for line in to_raw_text(text) for word in line]
    words = (" ".join(words).lower()).split()
    text = " ".join(words)
    
    for phrase in ["claim this business", "[ edit ]", "edit business info"]:
        phrase_pos = text.find(phrase)
        if phrase_pos != -1:
            words = text[phrase_pos + len(phrase):].split()
            text = " ".join(words)
    
    return words

def get_some_restaurants(num = 1000, collection_name = "restaurants", min_words = 0):
    global mongo_database_global
    texts_set = set()
    texts = []
    texts_data = []
    collection = mongo_database_global[collection_name]
    for index, post in enumerate(collection.find({}, {'categories': 1, 'saved_text': 1, "_id": 1, "id": 1, "price": 1, "rating": 1})):
        para = restaurant_saved_text_preprocess(post.get("saved_text"))
        if len(para) > min_words:
            para_line = " ".join(para)
            if para_line not in texts_set:
                texts.append(para)
                texts_data.append({'categories': post.get('categories'), "_id": post.get("_id"), "id":post.get("id"), "price":post.get("price"), "rating": post.get("rating")})
                texts_set.add(para_line)
            if index > num:
                break
    return (texts, texts_data)

def rating_to_string(rating):
    return "" + (int(rating) * "★")

def get_adjacency_matrix(nodes):
    adj = np.zeros([len(nodes), len(nodes)], dtype=np.float32)
    for node in nodes:
        if node.left != None:
            adj[node.id, node.left.id] = 1. - node.dist
            adj[node.left.id, node.id] = 1. - node.dist
            adj[node.id, node.right.id] = 1. - node.dist
            adj[node.right.id, node.id] = 1. - node.dist
    return adj

def assign_parents(node):
    if not hasattr(node, 'parent'):
        node.parent = None
    if node.left != None:
        node.left.parent = node
        assign_parents(node.left)
        node.right.parent = node
        assign_parents(node.right)
            
def get_degree_matrix(nodes):
    deg = np.zeros(len(nodes), dtype=np.float32)
    for node in nodes:
        if node.left != None:
            deg[node.id] += (1. - node.dist) * 2
            deg[node.left.id] += 1. - node.dist
            deg[node.right.id] += 1. - node.dist
    return deg

def norm_pdf(y, cov, inv = True):
    if inv:
        return np.exp(-0.5 * np.dot(np.dot(y.T, np.linalg.inv(cov)),y))
    else:
        return np.exp(-0.5 * np.dot(np.dot(y.T, cov),y))
    
def entropy(p_x):
    """
    Entropy:
    H(p_x) = - sum p_x * log(p_x)
    
    Inputs
    ------
    
    p_x ndarray : the probability distribution
    
    Outputs
    -------
    
    entropy float : the entropy of the distribution
    """
    return (-p_x * np.log(p_x)).sum() 


def present_restaurant(restaurant, text=None):
    if text is None:
        text = restaurant.get("saved_text").split(" ")
    display(HTML("""
    <div>
        <h2>%s</h2>
        <span style='color: #ca0814;'>%s</span><span style='color: #e4e4e4;'>%s</span><br /> 
        <span style='color: #feea60;'>%s</span><span style='color: #e4e4e4;'>%s</span>
        <span style='color:#333;font-size:9px'>%d words</span>
        <br/>
        <span style='color:#777:font-size:9px'>Categories: </span><span style='color:#333;font-size:13px'>%s</span>
        <p style='width:450px'>%s</p>
    </div>""" % (
        restaurant.get("_id"),
        "<span>" + "</span><span>".join(list(restaurant.get("price"))) + "</span>",
        "<span>$</span>" * (4-len(restaurant.get("price"))),
        rating_to_string(restaurant.get("rating")),
        rating_to_string(ceil(5 - restaurant.get("rating"))),
        len(text),
        ", ".join(restaurant.get('categories')),
        " ".join(text[20:50])
    )
    ))