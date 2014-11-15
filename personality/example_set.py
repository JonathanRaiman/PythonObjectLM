from utils import rating_to_string
import requests, pickle, gzip
from io import BytesIO
from lxml import html, etree
from .mtc import get_responses_for_hit

PAGES = {}

def save_example_sets(el,path):
    with gzip.open(path, "wb") as f:
        pickle.dump(el, f, pickle.HIGHEST_PROTOCOL)

def load_example_sets(path):
    file = gzip.open(path, "rb")
    sets = pickle.load(file)
    file.close()
    return sets

def obtain_html(url):
    if PAGES.get(url) is not None:
        tree = PAGES[url]
    else:
        page = requests.get(url)
        tree = html.fromstring(page.text)
        PAGES[url] = tree
    return tree

def review_to_snippet(review):
    description = review.cssselect("div.review-content > p")[0].text.split("\n")[0].strip()
    profile_pic = review.cssselect("img.photo-box-img")[0]
    #profile_name= review.cssselect("ul.user-passport-info")[0]
    
    return ("""<div class='row'>
    <div class='col_2'>%s</div>
    <div class='col_10'>“%s”</div>
    </div>""" % (tree_to_str(profile_pic), description))
    
def create_description_highlights(tree):
    
    reviews = tree.cssselect("div.review-list div.review")
    
    three_reviews = [review_to_snippet(review) for review in reviews[0:3]]
    
    return "<div><h3>Reviews:</h3>" + "".join(three_reviews) + "</div>"

def tree_to_str(tree):
    sio = BytesIO()
    doc = etree.ElementTree(tree)
    doc.write(sio)
    return sio.getvalue().decode("utf-8")

def clean_up_highlights(tree):
    links_in_highlight = tree.cssselect("p.quote a")
    
    ul_el = tree.cssselect("ul.review-highlights-list")
    for k, ul in enumerate(ul_el):
        if k==0:
            review_title = etree.Element("h3")
            review_title.text = "Reviews:"
            ul.getchildren()[0].addprevious(review_title)
        ul.tag = "div"
        ul.attrib["class"] = ""
        
    li_el = tree.cssselect("li.review-highlight")
    for li in li_el:
        li.tag = "div"
        li.attrib["class"] = "row"
        
    avatars = tree.cssselect(".media-avatar")
    for avatar in avatars:
        avatar.attrib["class"] = "col_2"
        
    media_stories = tree.cssselect(".media-story")
    for media_story in media_stories:
        media_story.attrib["class"] = "col_10"
    
    for a in links_in_highlight:
        a.tag = "span"
        a.attrib["href"]= ""
    
    pictures = tree.cssselect("img.photo-box-img")
    for piclink in pictures:
        piclink.getparent().tag = "span"
        piclink.getparent().attrib["href"] = ""
    
    sub_info = tree.cssselect("p.highlight-sub-info")
    for sub in sub_info:
        sub.getparent().remove(sub)
    return tree

class ExampleSet():
    
    def __init__(self, examples, options, personality_type = 0, difficulty = "easy", scraped = False):
        self.difficulty = difficulty
        self.personality_type = personality_type
        assert(len(examples) == 3), "Should provide 3 examples"
        assert(len(options) == 3),  "Should provide 3 options"
        if scraped:
            self.examples = examples
            self.options = options
        else:
            self.examples = [self.scrape_restaurant_data(example) for example in examples]
            self.options = [self.scrape_restaurant_data(option) for option in options]

    def get_responses(self, mtc):
        if hasattr(self, 'HITId'):
            return [response for hitid in self.HITId for response in get_responses_for_hit(mtc, hitid)]
        else:
            raise BaseException("No HITId for this ExampleSet")
        
    def scrape_restaurant_data(self, example):
        # get this from yelp
        
        html = obtain_html(example["url"])
        
        html.make_links_absolute(example["url"])
        
        title = html.cssselect("h1.biz-page-title")[0].text.strip()
        
        review_highlights = html.cssselect("ul.review-highlights-list")
        if len(review_highlights) > 0:
            description = tree_to_str(clean_up_highlights(review_highlights[0]))
        else:
            description = create_description_highlights(html)
        
        images = html.cssselect("img.photo-box-img")
        image_url = None
        if len(images) > 0:
            image_url   = images[0].attrib["src"]
        
        return {
        "title": title,
        "description": description,
        "categories": example["categories"],
        "image_url" : image_url,
        "rating": rating_to_string(example["rating"]),
        "price": example["price"]
        }
    
    def option_names(self):
        return [option["title"] for option in self.options]
    
    def example_names(self):
        return [example["title"] for example in self.examples]
        
    def example_to_html(self, ex):
        if ex["image_url"] is not None:
            return """<div class='row'>
                <div class="col_4">
                    <div class='restaurant_picture_frame'>
                        <img class='restaurant_picture' src='{image}'/>
                    </div>
                </div>
                <div class="col_8">
                    <h2>{title}</h2>
                    <span class='categories'>{categories}</span><br/>
                    <span class='rating'>{rating}</span> <span class='pricing'>{price}</span>
                </div></div><div>{description}</div>""".format(
                    title = ex["title"],
                    rating = ex["rating"],
                    price = ex["price"],
                    categories = ("<span class='category'>" + "</span>, <span class='category'>".join(ex["categories"])+"</span>"),
                    image = ex["image_url"],
                    description = ex["description"])
        else:
            return """<div class='row'>
                <div class="col_4">
                </div>
                <div class="col_8">
                    <h2>{title}</h2>
                    <span class='categories'>{categories}</span><br/>
                    <span class='rating'>{rating}</span> <span class='pricing'>{price}</span>
                </div></div><div>{description}</div>""".format(
                    title = ex["title"],
                    rating = ex["rating"],
                    categories = ("<span class='category'>" + "</span>, <span class='category'>".join(ex["categories"])+"</span>"),
                    price = ex["price"],
                    description = ex["description"])
        
    def get_examples_html(self):
        return [self.example_to_html(example) for example in self.examples]
    
    def get_options_html(self):
        return [self.example_to_html(example) for example in self.options]
        
    def to_html(self):
        html = []
        html_examples = self.get_examples_html()
        html_options  = self.get_options_html()
        
        message = "<div>Your friend <b>likes</b> these 3 restaurants : </div>"
        option_message = "<div>Among the following 3 restaurants, which would you recommend most:</div>"
        html.append(message)
        html.extend(html_examples)
        html.append(option_message)
        html.extend(html_options)
        
        return (html_examples, html_options)