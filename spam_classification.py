import glob
import re
import random
from collections import Counter
from naive_bayes_classifier import NaiveBayesClassifier

PATH = "./dataset/"
TP = (True, True) 
TN = (False, False)
FP = (True , False)
FN = (False , True)


def split_data(data, prob):
	random.shuffle(data)
	data_len = len(data)
	split = int(prob * data_len)
	train_data = data[:split]
	test_data = data[split:]
	return train_data, test_data

def spammiest_word(classifier):
	words_with_prob = classifier.word_prob
	words_with_prob.sort(key = lambda row:row[1]/(row[1]+row[2]))
	return [w[0] for w in words_with_prob][-5:]

def accuracy(result):
	return (result[TP] + result[TN])/(result[TP]+result[TN]+result[FP]+result[FN])
	
def precision(result):
	return result[TP]/(result[TP]+result[FP])

def recall(result):
	return (result[TP]/(result[TP]+result[FN]))

def most_misclassified(result):
	# sorting in acending order of spam probability
	result.sort(key=lambda row:row[2])
	#have high probability of being classified as spam while it is not spam
	spammiest_hams = list(filter(lambda row:not row[1], result))[-5:]

	#have lowest probability of beign classified as spam while it is spam
	hammiest_spams = list(filter(lambda row:row[1], result))[:5]

	return spammiest_hams,hammiest_spams

def main():
	data = []
	for verdict in ['spam', 'not_spam']:
		for files in glob.glob(PATH + verdict + "/*")[:500]:
			is_spam = True if verdict == 'spam' else False
			with open(files,"r",encoding='utf-8', errors='ignore') as f:
				for line in f:
					if line.startswith("Subject:"):
						subject = re.sub("^Subject: ", "", line).strip()
						data.append((subject, is_spam))
	
	random.seed(0)
	train_data, test_data = split_data(data, 0.75)
	classifier = NaiveBayesClassifier()
	classifier.train(train_data)

	print("Spam" if classifier.classify("Get free laptops now!")>0.5 else "Not Spam")

	classified = [(subject, is_spam, classifier.classify(subject))
				  for subject, is_spam in test_data]

	count = Counter((is_spam, spam_probability > 0.5)
					for _, is_spam, spam_probability in classified)

	spammiest_hams, hammiest_spams = most_misclassified(classified)

	print("Accuracy: ", accuracy(count))
	print("Precision: ", precision(count))
	print("Recall: ", recall(count))
	print("\nTop 5 falsely classified as spam:\n\n",spammiest_hams)
	print("\nTop 5 falsely classified as not spam:\n\n",hammiest_spams)
	print("\nMost spammiest words: ",spammiest_word(classifier))

if __name__ == "__main__":
main()
