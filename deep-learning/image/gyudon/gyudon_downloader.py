import sys, os, re, time
import urllib.request as req
import urllib.parse as parse
import json, urllib

# API URL
PHOTOZOU_API = "https://api.photozou.jp/rest/search_public.json"
CACHE_DIR = "./image/cache"

# search images using api
def search_photo(keyword, offset=0, limit=100):
  # api query
  keyword_enc = parse.quote_plus(keyword)
  q = "keyword={0}&offset={1}&limit={2}".format(keyword_enc, offset, limit)
  # url = PHOTOZOU_API + "?" + q
  url = "https://api.photozou.jp/rest/search_public.json?%s" % parse.quote(q)

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

# download image
def download_thumb(info, save_dir):
  if not os.path.exists(save_dir): os.makedirs(save_dir)
  if info is None: return
  if not "photo" in info["info"]:
    print("[ERROR] broken info")
    return
  photolist = info["info"]["photo"]
  for photo in photolist:
    title = photo["photo_title"]
    photo_id = photo["photo_id"]
    url = photo["thumbnail_image_url"]
    path = save_dir + "/" + str(photo_id) + "_thumb.jpg"
    if os.path.exists(path): continue
    try:
      print("[download]", title, photo_id)
      req.urlretrieve(url, path)
      time.sleep(1)
    except Exception as e:
      print("[ERROR] failed to download url=", url)

# search all and download
def download_all(keyword, save_dir, maxphoto=1000):
  offset = 0
  limit = 100
  while True:
    # call API
    info = search_photo(keyword, offset=offset, limit=limit)
    if info is None:
      print("[ERROR] no result"); return
    if (not "info" in info) or (not "photo_num" in info["info"]):
      print("[ERROR] broken data"); return
    photo_num = info["info"]["photo_num"]
    if photo_num == 0:
      print("photo_num = 0, offset=", offset)
      return
    
    print("*** download offset=", offset)
    download_thumb(info, save_dir)
    offset += limit
    if offset >= maxphoto: break

if __name__ == '__main__':
  download_all("牛丼", "./image/gyudon")
