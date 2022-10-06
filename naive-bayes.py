# File Author: Mikkel Folting
# Date: October 7, 2022

# Results
  # Precision:       83.2%
  # Recall:          79.2%
  # F-Score:          0.81
# How I handled tokens
  # My first step in handling tokens was to remove all stop words. To do this, I used the nltk stopwords list.
  # Later, I ignore words in the test set that we have never seen before because we don't know how to classify them.
# What smoothing I used
  # I used Laplace (add-one) smoothing to avoid 0 probabilities when we encountered words that were in one training set but not the other.
# Any other tricks
  # I didn't use any other significant tricks to get to my results.

import os
import re
import math
import nltk.lm
from nltk.corpus import stopwords

# UNCOMMENT BEFORE RUNNING FOR THE FIRST TIME
# nltk.download('stopwords')

posTrain = {}     # A dictionary containing the counts of each distinct word in the positive training set
posWordCount = 0  # Total number of non-distinct words found in the positive training set
negTrain = {}     # A dictionary containing the counts of each distinct word in the positive training set
negWordCount = 0  # Total number of non-distinct words found in the negative training set
vocab = set()     # The set of every distinct word encountered in training
posTest = []      # List of filenames for positive reviews to test on
negTest = []      # List of filenames for negative reviews to test on

numPosReviews = 0 # Stores the amount of positive reviews we read for training
numNegReviews = 0 # Stores the amount of negative reviews we read for training

posDirectory = 'movie_reviews/pos'  # Directory to get to positive reviews
negDirectory = 'movie_reviews/neg'  # Directory to get to negative reviews

def createTrainAndTestSets(directory, filename, count, stopWords):
  global posWordCount, negWordCount

  # Get current file and store entire corpus
  f = os.path.join(directory, filename)
  file = open(f)
  corpus = file.read()
  file.close()

  # Pre-process the corpus, removing unnecessary characters
  corpus = re.sub("\n", " ", corpus)                    # Remove newline characters
  corpus = re.sub(r'-+', " ", corpus)                   # Remove any occurrences of - and replace with a space
  corpus = re.sub(r'[\.?!,;():`_*"\']', "", corpus)     # Remove punctuation & symbols
  words = list(nltk.tokenize.word_tokenize(corpus))     # Tokenize words

  # Filter out stop words from our list of words
  filteredWords = []
  for w in words:
    if w not in stopWords and w != '\'s':
      filteredWords.append(w)

  # We no longer need our words list so we can delete it
  del words

  # Reserve 25% of the reviews for testing and exit this function
  if count % 4 == 0:
    if directory == posDirectory:
      posTest.append(filteredWords)
    else:
      negTest.append(filteredWords)
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

  # For all positive reviews, build a train and test set
  for filename in os.listdir(posDirectory):
    createTrainAndTestSets(posDirectory, filename, count, stopWords)
    count = count + 1
  numPosReviews = count   # Store the total amount of positive reviews
  count = 0
  print("Training and test set for positive reviews created!")

  # For all negative reviews, build and train a test set
  for filename in os.listdir(negDirectory):
    createTrainAndTestSets(negDirectory, filename, count, stopWords)
    count = count + 1
  numNegReviews = count   # Store the total amount of negative reviews
  print("Training and test set for negative reviews created!")
  return

def calculateResults(truePos, falsePos, trueNeg, falseNeg):
  # Calculate precision, recall, and f-score based on our classifications and print to the console
  precision = truePos / (truePos + falsePos)
  precisionPercentage = "{:.1%}".format((precision))
  recall = truePos / (truePos + falseNeg)
  recallPercentage = "{:.1%}".format((recall))
  fscore = (2 * precision * recall) / (precision + recall)
  print("______________________")
  print("RESULTS")
  print("Precision:\t", precisionPercentage)
  print("Recall: \t", recallPercentage)
  print("F-Score:\t%6.2f\n" % (fscore))

  print("Confusion matrix:")
  print(str(truePos) + "\t" + str(falsePos))
  print(str(falseNeg) + "\t" + str(trueNeg))

def testClassifiers():  
  # Parse through all positive test reviews
  truePos = 0
  falseNeg = 0
  print("Number of positive test reviews:", len(posTest))
  print("Number of negative test reviews:", len(negTest))
  for review in posTest:
    productPos = math.log((float(numPosReviews)) / (numPosReviews + numNegReviews))
    productNeg = math.log((float(numNegReviews)) / (numPosReviews + numNegReviews))
    for w in review:
      # If the current word isn't in our vocabulary, we will simply ignore it and move on
      if w not in vocab:
        continue
      
      # Find how many times given word occurs in the positive and negative training sets
      # Default will be zero if they aren't found
      posCount = 0
      negCount = 0
      if w in posTrain:
        posCount = posTrain.get(w)
      if w in negTrain:
        negCount = negTrain.get(w)

      # Use add-one smoothing and add to the total probability of positive and negative classification
      productPos = productPos + math.log(((posCount + 1) / (posWordCount + len(vocab))))
      productNeg = productNeg + math.log(((negCount + 1) / (negWordCount + len(vocab))))
    # Attribute the highest probability to a true positive or false negative depending on which probability is highest 
    if productPos >= productNeg:
      truePos = truePos + 1
    else:
      falseNeg = falseNeg + 1

  # Parse through all negative test reviews
  falsePos = 0
  trueNeg = 0
  for review in negTest:
    productPos = math.log((float(numPosReviews)) / (numPosReviews + numNegReviews))
    productNeg = math.log((float(numNegReviews)) / (numPosReviews + numNegReviews))
    for w in review:
      # If the current word isn't in our vocabulary, we will simply ignore it and move on
      if w not in vocab:
        continue

      # Find how many times given word occurs in the positive and negative training sets
      # Default will be zero if they aren't found
      posCount = 0
      negCount = 0
      if w in posTrain:
        posCount = posTrain.get(w)
      if w in negTrain:
        negCount = negTrain.get(w)
      
      # Use add-one smoothing and add to the total probability of positive and negative classification
      productPos = productPos + math.log(((posCount + 1) / (posWordCount + len(vocab))))
      productNeg = productNeg + math.log(((negCount + 1) / (negWordCount + len(vocab))))
    # Attribute the highest probability to a false positive or true negative depending on which probability is highest 
    if productPos > productNeg:
      falsePos = falsePos + 1
    else:
      trueNeg = trueNeg + 1
  
  # Send control to a function to calculate and print our results!
  calculateResults(truePos, falsePos, trueNeg, falseNeg)
  return

def main():
  # First we will create the sets of counts for positive and negative reviews
  print("Creating training and test sets...")
  createPosAndNegSets()

  # Now that we have the sets created, we want to start looking at the test sets to find probabilities
  print("Testing our classifiers...")
  testClassifiers()
  return

if __name__ == '__main__':
  main()