import re
import math
from collections import defaultdict, Counter


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_prob = []

    def train(self, training_set):
        total_spams = total_non_spams = 0
        for message, verdict in training_set:
            if verdict:
                total_spams += 1
            else:
                total_non_spams += 1
        counts = self.count_words(training_set)
        self.word_prob = self.word_probabilities(counts,total_spams,total_non_spams)

    def classify(self, message):
        return self.spam_probability(message)

    def tokenize(self, message):
        message = message.lower()
        all_words = re.findall("[a-z0-9]+", message)
        return set(all_words)

    def count_words(self, training_set):
        "Trainig set consist of pairs (message, is_spam)"
        counts = defaultdict(lambda: [0, 0])
        for message, is_spam in training_set:
            for word in self.tokenize(message):
                counts[word][0 if is_spam else 1] += 1
        return counts

    def word_probabilities(self, counts, total_spams, total_non_spams):
        """turns the word_count into list of triplets [w, P(w/spam), P(w/~spam)]
        It gives the probability of word being in spam and not being in spam
        P(w|S) = (k + number of spam containing w)/(2*k + total number of spams)
        P(w |~S) = (k + number of ~spam containing w)/(2*k + total number of ~spams)"""
        return [(w,
				(self.k + spam) / (2 * self.k + total_spams),
				(self.k + not_spam) / (2 * self.k + total_non_spams))
				for w,(spam,not_spam) in counts.items()]

    def spam_probability(self,message):
        message_words = self.tokenize(message)
        log_prob_if_spam = log_prob_if_not_spam = 0.0

        for word, prob_if_spam, prob_if_not_spam in self.word_prob:
            if word in message_words:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_not_spam += math.log(prob_if_not_spam)

            # if *word* doesn't appear in the message
            # add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_not_spam = math.exp(log_prob_if_not_spam)
        
        # P(spam|x1,x2,x3..xi) = P(x1,x2..xi|S) / [P(x1,x2..xi|S) + P(x1,x2..xi|~S)]
        # if P(S) == P(~S)
return prob_if_spam / (prob_if_spam + prob_if_not_spam)
