import sys, os, re, time
import urllib.request as req
import urllib.parse as parse
import json

# API URL
PHOTOZOU_API = "https://api.photozou.jp/rest/search_public.json"
CACHE_DIR = "./image/cache"

# search images using api
def search_photo(keyword, offset=0, limit=100):
  # api query
  keyword_enc = parse.quote_plus(keyword)
  q = "keyword={0}&offset={1}&limit={2}".format(keyword_enc, offset, limit)
  url = PHOTOZOU_API + "?" + q

  # make a folder for caching
  if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
  cache = CACHE_DIR + "/" + re.sub(r'[^a-zA-Z0-9\%\#]+', '_', url)
  if os.path.exists(cache):
    return json.load(open(cache, "r", encoding="utf-8"))
  print("[API] " + url)
  req.urlretrieve(url, cache)
  time.sleep(1)
  return json.load(open(cache, "r", encoding="utf-8"))
