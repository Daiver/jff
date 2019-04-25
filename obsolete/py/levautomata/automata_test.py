import automata
import bisect
import random
 
 
class Matcher(object):
  def __init__(self, l):
    self.l = l
    self.probes = 0
  def __call__(self, w):
    self.probes += 1
    pos = bisect.bisect_left(self.l, w)
    if pos < len(self.l):
      return self.l[pos]
    else:
      return None
 
 
words = [x.strip().lower().decode('utf-8') for x in open('/usr/share/dict/words')]
print 'l', len(words)
#words = [x.strip().lower().decode('utf-8') for x in open('/usr/share/dict/web2')]
words.sort()
words10 = [x for x in words if random.random() <= 0.1]
words100 = [x for x in words if random.random() <= 0.01]
 
 
m = Matcher(words)
from time import time
print  len(list(automata.find_all_matches('food', 1, m))) 
st = time()
print  (list(automata.find_all_matches('mosco', 2, m))) 
print 'st', time() - st
print m.probes
assert len(list(automata.find_all_matches('food', 1, m))) == 18
print m.probes
 
m = Matcher(words)
assert len(list(automata.find_all_matches('food', 2, m))) == 283
print m.probes
 
 
def levenshtein(s1, s2):
  if len(s1) < len(s2):
    return levenshtein(s2, s1)
  if not s1:
    return len(s2)
  
  previous_row = xrange(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
      deletions = current_row[j] + 1     # than s2
      substitutions = previous_row[j] + (c1 != c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row
  
  return previous_row[-1]
 
class BKNode(object):
  def __init__(self, term):
    self.term = term
    self.children = {}
  
  def insert(self, other):
    distance = levenshtein(self.term, other)
    if distance in self.children:
      self.children[distance].insert(other)
    else:
      self.children[distance] = BKNode(other)
  
  def search(self, term, k, results=None):
    if results is None:
      results = []
    distance = levenshtein(self.term, term)
    counter = 1
    if distance <= k:
      results.append(self.term)
    for i in range(max(0, distance - k), distance + k + 1):
      child = self.children.get(i)
      if child:
        counter += child.search(term, k, results)
    return counter
