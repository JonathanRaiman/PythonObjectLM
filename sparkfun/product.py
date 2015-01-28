class Product(object):
    __slots__ = ["sku", "name", "description", "categories", "price"]
    
    def __init__(self, sku, name, description, categories, price):
        self.sku = sku
        self.name = name
        self.description = description
        self.categories = categories
        self.price = price