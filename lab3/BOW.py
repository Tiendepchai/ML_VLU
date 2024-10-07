import numpy as np
from collections import defaultdict 
 
#Sample text corpus
data = ['She loves pizza, pizza is delicious.','She is a good person.','good people are the best.']
 
#clean the corpus.
lower_case_documents=list(map(lambda x:x.lower(),data))
sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(''.join(c for c in i if c not in string.punctuation))
    
preprocessed_documents=[i.split(' ') for i in sans_punctuation_documents]
dictionary = defaultdict(int)
for i in preprocessed_documents:
  for j in i:
     dictionary[j] = 0   
print(dictionary)