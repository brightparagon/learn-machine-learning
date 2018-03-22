import math, sys
from konlpy.tag import Twitter

class BayesianFilter:
  """ Bayesian Filter"""
  def __init__(self):
    self.words = set() # shown words
    self.word_dict = {} # shown words for category
    self.category_dict = {} # shown category

  # analyze morphemes
  def split(self, text):
    results = []
    twitter = Twitter()

    malist = twitter.pos(text, norm=True, stem=True)
    for word in malist:
      # ignore those morphemes
      if not word[1] in ["Josa", "Eomi", "Punctuation"]:
        results.append(word[0])
    
    return results

  # count appearance of word and category
  def inc_word(self, word, category):
    if not category in self.word_dict:
      self.word_dict[category] = {}
    if not word in self.word_dict[category]:
      self.word_dict[category][word] = 0
    self.word_dict[category][word] += 1
    self.words.add(word)

  def inc_category(self, category):
    if not category in self.category_dict:
      self.category_dict[category] = 0
    self.category_dict[category] += 1

  # learn texts
  def fit(self, text, category):
    word_list = self.split(text)
    for word in word_list:
      self.inc_word(word, category)
    self.inc_category(category)

  # score
  def score(self, words, category):
    score = math.log(self.category_prob(category))
    for word in words:
      score += math.log(self.word_prob(word, category))
    return score

  # predict
  def predict(self, text):
    best_category = None
    max_score = -sys.maxsize
    words = self.split(text)
    score_list = []
    for category in self.category_dict.keys():
      score = self.score(words, category)
      score_list.append((category, score))
      if score > max_score:
        max_score = score
        best_category = category

    return best_category, score_list

  # count word in category
  def get_word_count(self, word, category):
    if word in self.word_dict[category]:
      return self.word_dict[category][word]
    else:
      return 0

  # calculate category
  def category_prob(self, category):
    sum_categories = sum(self.category_dict.values())
    category_v = self.category_dict[category]
    return category_v / sum_categories

  # calculate word in category
  def word_prob(self, word, category):
    n = self.get_word_count(word, category) + 1
    d = sum(self.word_dict[category].values()) + len(self.words)
    return n / d
  