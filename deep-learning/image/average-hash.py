from PIL import Image
import numpy as np

# convert image to Average Hash
def average_hash(fname, size = 16):
  img = Image.open(fname) # open image data
  img = img.convert('L') # change to grey scale
  img = img.resize((size, size), Image.ANTIALIAS) # resize with an option of anti alias
  pixel_data = img.getdata() # get pixel data
  pixels = np.array(pixel_data) # # change to numpy array
  pixels = pixels.reshape((size, size)) # change to 2 dim array
  avg = pixels.mean() # get average
  diff = 1 * (pixels > avg) # if greater than avg 1, if not 0
  return diff

# convert to binary hash
def np2hash(ahash):
  bhash = []
  for nl in ahash.tolist():
    sl = [str(i) for i in nl]
    s2 = "".join(sl)
    i = int(s2, 2) # convert binary to integer
    bhash.append("%04x" % i)
  return "".join(bhash)

# print Average Hash
ahash = average_hash('operahouse.jpg')
print(ahash)
print(np2hash(ahash))
