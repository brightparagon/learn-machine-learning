import cgi, os.path
from sklearn.externals import joblib

# read learned data
pklfile = os.path.dirname(__file__) + "./lang/freq.pkl"
clf = joblib.load(pklfile)

# html that gets text from input tags
def show_form(text, msg=""):
  print("Content-Type: text/html; charset=utf-8")
  print("")
  print("""
    <html><body><form>
    <textarea name="text" rows="8" cols="40">{0}</textarea>
    <p><input type="submit" value="distinguish"></p>
    <p>{1}</p>
    </form></body></html>
  """.format(cgi.escape(text), msg))

# distinguish
def detect_language(text):
  text = text.lower()
  code_a, code_z = (ord("a"), ord("z"))
  cnt = [0 for i in range(26)]
  for ch in text:
    n = ord(ch) - code_a
    if 0 <= n < 26: cnt[n] += 1

  total = sum(cnt)
  if total == 0: return "There's no input"
  freq = list(map(lambda n: n/total, cnt))

  # predict a language
  res = clf.predict([freq])
  # convert language code to English
  lang_dic = {"en": "English", "fr": "French", "id": "Indonesian", "tl": "Tagalog"}
  return lang_dic[res[0]]

# read values from form
form = cgi.FieldStorage()
text = form.getvalue("text", default="")
msg = ""
if text != "":
  lang = detect_language(text)
  msg = "result: " + lang

# print the result
show_form(text, msg)
