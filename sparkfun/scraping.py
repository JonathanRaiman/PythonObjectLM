import requests
from lxml import html, etree
from io import BytesIO

PAGES = {}
def obtain_html(url):
    if PAGES.get(url) is not None:
        tree = PAGES[url]
    else:
        page = requests.get(url)
        tree = html.fromstring(page.text)
        PAGES[url] = tree
    return tree

def sparkfun_sub_urls(seed_url):
    html = obtain_html(seed_url)
    product_tiles = [a.attrib["href"] for a in html.cssselect(".product-tile .title a[itemprop='url']")]
    return product_tiles
    
def sparkfun_seed_urls(seed_url):
    html = obtain_html(seed_url)
    categories = [a.attrib["href"] if a.attrib["href"].startswith("http") else seed_url + a.attrib["href"]  for a in html.cssselect("ul#category_menu_list > li > a") if len(a.text) > 2 and a.attrib["href"].find("categories") > -1]
    return categories

from .product import Product

def sparkfun_html_meta(url):
    html = obtain_html(url)
    description = html.cssselect("div.col-description .description")[0].text_content().strip()
    categories = [text for text in map(lambda x: x.text_content(), html.cssselect(".is_open.current > a")) if len(text) > 2]
    try:
        price = float(html.cssselect(".pricing span[itemprop='price']")[0].text)
    except ValueError:
        price = float(html.cssselect(".pricing span[itemprop='price']")[0].text.replace(",", ""))
    except IndexError:
        return None
    name = html.cssselect("h1[itemprop='name']")[0].text
    
    sku = html.cssselect("meta[itemprop='sku']")[0].attrib["content"]
    return Product(sku, name, description, categories, price)

def scrape_pages(verbose=True):
    products = {}
    seed_urls = sparkfun_seed_urls("https://www.sparkfun.com")
    for k, seed_url in enumerate(seed_urls):
        product_pages = sparkfun_sub_urls(seed_url)
        if verbose:
            print("seed %d / %d : %d products" % (k, len(seed_urls), len(product_pages)))
        for product_page in product_pages:
            product = sparkfun_html_meta(product_page)
            if product is not None and product.sku not in products:
                products[product.sku] = product

    return products