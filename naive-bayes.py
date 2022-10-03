import sys
import os
import re
import nltk.lm
from nltk.corpus import stopwords

# UNCOMMENT BEFORE RUNNING FOR THE FIRST TIME
# nltk.download('stopwords')

posTrain = {}
negTrain = {}
vocab = set()
testCorpuses = []

posDirectory = 'movie_reviews/pos'
negDirectory = 'movie_reviews/neg'

def createTrainAndTestSets(directory, filename, count, stopWords):
  # if count > 15:
  #   return 
  f = os.path.join(directory, filename)
  file = open(f)
  corpus = file.read()
  file.close()

  corpus = re.sub("\n", " ", corpus) # Remove newline characters
  corpus = re.sub(r'-+', " ", corpus) # Remove any occurrences of - and replace with a space
  corpus = re.sub(r'[\.?!,;():`"]', "", corpus) # Remove punctuation and parantheses
  words = list(nltk.tokenize.word_tokenize(corpus)) # Tokenize words

  # Filter out stop words from our list of words
  filteredWords = []
  for w in words:
    if w not in stopWords and w != '\'s':
      filteredWords.append(w)

  # Reserve 25% of the reviews for testing
  if count % 4 == 0:
    testCorpuses.append(words)
    return

  # Add every word we see in our filtered words list to our vocab
  for w in filteredWords:
    # Add any new words we encounter to our vocabulary
    if w not in vocab:
      vocab.add(w)
    
    # Add the words into the approprate training set and increment counts
    if directory == posDirectory:
      if w in posTrain:
        val = posTrain.get(w)
        posTrain.update({w : val + 1})
      else:
        posTrain.update({w : 1})
    else:
      if w in negTrain:
        val = negTrain.get(w)
        negTrain.update({w : val + 1})
      else:
        negTrain.update({w : 1})
  return

def createPosAndNegSets():
  # Set of stop words to be removed from the corpus
  stopWords = set(stopwords.words('english'))
  count = 0
  for filename in os.listdir(posDirectory):
    createTrainAndTestSets(posDirectory, filename, count, stopWords)
    count = count + 1
  count = 0
  for filename in os.listdir(negDirectory):
    createTrainAndTestSets(negDirectory, filename, count, stopWords)
    count = count + 1
  return

def testClassifiers():
  

def main():
  # First we will create the sets of counts for positive and negative reviews
  createPosAndNegSets()

  # Now that we have the sets created, we want to start looking at the test sets to find probabilities
  testClassifiers()
  return

if __name__ == '__main__':
  main()