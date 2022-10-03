import sys
import os
import re
import nltk.lm
from nltk.corpus import stopwords

# UNCOMMENT BEFORE RUNNING FOR THE FIRST TIME
# nltk.download('stopwords')

posTrain = {}
posWordCount = 0
negTrain = {}
negWordCount = 0
vocab = set()
posTest = []
negTest = []

numPosReviews = 0
numNegReviews = 0

posDirectory = 'movie_reviews/pos'
negDirectory = 'movie_reviews/neg'

def createTrainAndTestSets(directory, filename, count, stopWords):
  global posWordCount, negWordCount
  f = os.path.join(directory, filename)
  file = open(f)
  corpus = file.read()
  file.close()

  corpus = re.sub("\n", " ", corpus) # Remove newline characters
  corpus = re.sub(r'-+', " ", corpus) # Remove any occurrences of - and replace with a space
  corpus = re.sub(r'[\.?!,;():`"\']', "", corpus) # Remove punctuation and parantheses
  words = list(nltk.tokenize.word_tokenize(corpus)) # Tokenize words

  # Filter out stop words from our list of words
  filteredWords = []
  for w in words:
    if w not in stopWords and w != '\'s':
      filteredWords.append(w)

  # Reserve 25% of the reviews for testing
  if count % 4 == 0:
    if directory == posDirectory:
      posTest.append(words)
    else:
      negTest.append(words)
    return

  # Add every word we see in our filtered words list to our vocab
  for w in filteredWords:
    # Add any new words we encounter to our vocabulary
    if w not in vocab:
      vocab.add(w)
    
    # Add the words into the approprate training set and increment counts
    if directory == posDirectory:
      posWordCount = posWordCount + 1
      if w in posTrain:
        val = posTrain.get(w)
        posTrain.update({w : val + 1})
      else:
        posTrain.update({w : 1})
    else:
      negWordCount = negWordCount + 1
      if w in negTrain:
        val = negTrain.get(w)
        negTrain.update({w : val + 1})
      else:
        negTrain.update({w : 1})
  return

def createPosAndNegSets():
  global numPosReviews, numNegReviews
  # Set of stop words to be removed from the corpus
  stopWords = set(stopwords.words('english'))
  count = 0
  for filename in os.listdir(posDirectory):
    createTrainAndTestSets(posDirectory, filename, count, stopWords)
    count = count + 1
  numPosReviews = count
  count = 0
  for filename in os.listdir(negDirectory):
    createTrainAndTestSets(negDirectory, filename, count, stopWords)
    count = count + 1
  numNegReviews = count
  return

def testClassifiers():
  totalPos = len(posTest)
  totalNeg = len(negTest)
  
  truePos = 0
  falsePos = 0
  for review in posTest:
    productPos = (float(numPosReviews)) / (numPosReviews + numNegReviews)
    productNeg = (float(numNegReviews)) / (numPosReviews + numNegReviews)
    for w in review:
      # FIX - ADD SMOOTHING TO PREVENT ERROR WHERE WORD DOESN'T APPEAR IN ONE OF THE TRAINING SETS
      # if w not in vocab:
      #   continue
      if w not in vocab:
        continue

      posCount = 0
      negCount = 0
      if w in posTrain:
        posCount = posTrain.get(w)
      if w in negTrain:
        negCount = negTrain.get(w)

      productPos = productPos * ((posCount + 1) / (posWordCount + len(vocab))) 
      productNeg = productNeg * ((negCount + 1) / (negWordCount + len(vocab)))
    print(productPos, productNeg)
    if productPos >= productNeg:
      truePos = truePos + 1
    else:
      falsePos = falsePos + 1

  trueNeg = 0
  falseNeg = 0
  for review in negTest:
    productPos = (float(numPosReviews)) / (numPosReviews + numNegReviews)
    productNeg = (float(numNegReviews)) / (numPosReviews + numNegReviews)
    for w in review:
      # FIX - ADD SMOOTHING TO PREVENT ERROR WHERE WORD DOESN'T APPEAR IN ONE OF THE TRAINING SETS
      # if w not in vocab:
      #   continue

      if w not in vocab:
        continue

      posCount = 0
      negCount = 0
      if w in posTrain:
        posCount = posTrain.get(w)
      if w in negTrain:
        negCount = negTrain.get(w)

      productPos = productPos * ((posCount + 1) / (posWordCount + len(vocab))) 
      productNeg = productNeg * ((negCount + 1) / (negWordCount + len(vocab)))
    if productPos > productNeg:
      falseNeg = falseNeg + 1
    else:
      trueNeg = trueNeg + 1

  print(str(truePos) + "\t" + str(falsePos))
  print(str(falseNeg) + "\t" + str(trueNeg))

  precision = truePos / (truePos + falsePos)
  precisionPercentage = "{:.1%}".format((precision))
  recall = truePos / (truePos + falseNeg)
  recallPercentage = "{:.1%}".format((recall))
  fscore = (2 * precision * recall) / (precision + recall)
  print("Precision:\t", precisionPercentage)
  print("Recall: \t", recallPercentage)
  print("F-Score:\t", fscore)

def main():
  # First we will create the sets of counts for positive and negative reviews
  print("Creating training and test sets...")
  createPosAndNegSets()
  print("Positive word count:", posWordCount)
  print("Negative word count:", negWordCount)
  # Now that we have the sets created, we want to start looking at the test sets to find probabilities
  print("Testing our classifiers...")
  testClassifiers()
  return

if __name__ == '__main__':
  main()