import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from textblob import TextBlob
# !pip install contractions
import contractions
from copy import deepcopy
import torch


class S2STextPreprocessor():
	"""
			This class is used to preprocess a corpus (a list of 
		pairs of strings (texts)) for training a chatbot. 
        Preprocessing contains cleansing the texts and building 
        data structures needed in the training.
			The methods provided to cleanse the texts include:
			(1) Chop off all characters that are absent in a 
			defined library.
			(2) Lower the case of all characters.
			(3) Apply a customized mapping to each text string via 
			a provided map.
			(4) Expand all contractions.
			(5) Fix the misspellings.
			(6) Add space on both sides of all punctuations for 
            further tokenization.
			The data structures that can be constructed by the 
		provided methods include:
			(1) Perform word-level tokenization on a text corpus 
			to get a corresponding corpus in the form of tokens 
			(Two lists of lists of tokens).
			(2) Build a sorted vocabulary (sorted_vocab) containing 
			(word, num_word) pairs for all words (list of tuples).
            Using the sorted_vocab, the corpus can be trimmed to have 
            only input-output pairs that do not contain any word out
            of a specified number of the most frequently used words.
			(3) Build a word2idx map mapping each word to an integer.
			(4) Build two lists containing integer sequences with each 
			sequence stored as a sublist. Each integer is mapped from 
			a token via the word2idx map.
			(5) The sequences in one mini-batch can be all padded with
            0 from either the front or the end to match the length of
            the longest sequence. The padding are performed to the 
            input and output sequences in a mini-batch independently.
			(6) Build a embedding matrix.
	"""

	def __init__(self, max_num_words = None,
				 minimum_count = None,
				 max_seq_length = None,
				 front_padded = False,
				 chop = False,
				 lower = True,
				 contracted = False,
				 fix_misspelling = False,
				 to_keep = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
				 customized_map = None,
                 trimming = None):
			
		self.max_num_words = max_num_words
		self.minimum_count = minimum_count
		self.max_seq_length = max_seq_length
		self.front_padded = front_padded
		self.chop = chop
		self.lower = lower
		self.contracted = contracted
		self.fix_misspelling = fix_misspelling
		self.to_remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
		self.to_keep = to_keep
		self.customized_map = customized_map
		self.character_lib = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
				 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
				 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
				 '!', '"', '\'', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', 
				 '^', '_', '`', '{', '|', '}', '~', '\t', '\n', ' ', '', "’", "‘", "´", "`")
		self.tokenized_corpus = list()
		self.sorted_vocab = list()
		self.word2idx = {"PAD": 0, "SOS": 1, "EOS": 2}
		self.idx2word = {0: "PAD", 1: "SOS", 2: "EOS"}
		self.sequences = list()
		self.num_words = int()
		self.num_tokens = int()
		self.trimming = trimming

		for punc in self.to_keep:
			self.to_remove = self.to_remove.replace(punc, '')

	def cleanse_corpus(self, corpus):
		"""
				Cleanse the corpus.
		"""
		corpus = pd.DataFrame(corpus)

		if self.chop:
			corpus = corpus.applymap(lambda x: self.chop_off(x))
		if self.customized_map:
			corpus = corpus.applymap(lambda x: self.customized_cleansing(x))
		if not self.contracted:
			corpus = corpus.applymap(lambda x: self.expand_contractions(x))
		if self.lower:  # Place lower_case() after expand_contractions() because contractions.fit() doesn't preserve the case of some words
			corpus = corpus.applymap(lambda x: self.lower_case(x))
		if self.fix_misspelling:
			corpus = corpus.applymap(lambda x: self.corret_misspelling(x))
		corpus = corpus.applymap(lambda x: self.process_punctuations(x))

		return corpus

	def chop_off(self, text):
		for character in text:
			if character not in self.character_lib:
				text = text.replace(character, ' ')
		return text

	def lower_case(self, text):
		text = text.lower()
		return text

	def expand_contractions(self, text):
		for item in ["’", "‘", "´", "`"]:
			text = text.replace(item, "'")
		try:
			text = contractions.fix(text)
		except IndexError:
			pass
		return text

	def corret_misspelling(self, text):
		text = str(TextBlob(text).correct())
		return text

	def customized_cleansing(self, text):
		for item in self.customized_map:
			text = text.replace(item, self.customized_map[item])
		return text

	def process_punctuations(self, text):
		for punc in self.to_keep:
			text = text.replace(punc, f' {punc} ')

		for punc in self.to_remove:
			text = text.replace(punc, ' ')
		return text	

	def fit_on_corpus(self, corpus):
		"""
				Fit the tokenizer on a corpus, which is 
			a DataFrame of two columns (of texts) containing
			inputs and outputs, respectively.
		"""
		inputs = self.tokenize_corpus(corpus.iloc[:, 0])
		outputs = self.tokenize_corpus(corpus.iloc[:, 1])
		self.tokenized_corpus = (inputs, outputs)
  
		self.sorted_vocab = self.build_sorted_vocab(self.tokenized_corpus)
  
		self.word2idx.update(self.build_word2idx(self.sorted_vocab))
		self.idx2word.update(self.build_idx2word(self.sorted_vocab))
  
		if self.trimming:
			self.tokenized_corpus = self.trim_corpus(self.tokenized_corpus)
			assert len(self.tokenized_corpus[0]) > 0, "Corpus is empty, trimmed too much!"

		inputs = self.texts_to_sequences(self.tokenized_corpus[0], self.word2idx)
		outputs = self.texts_to_sequences(self.tokenized_corpus[1], self.word2idx)
		self.sequences = (inputs, outputs)

	def tokenize_corpus(self, corpus): 
		return [[j for j in i.split() if j] for i in corpus]

	def build_sorted_vocab(self, tokenized_corpus):
        # Build a sorted vocabulary containing all words
		vocab = defaultdict(int)
		for col in tokenized_corpus:
			for text in col:
				for token in text:
					vocab[token] += 1
		sorted_vocab = list(vocab.items())
		sorted_vocab.sort(key=lambda x: x[1], reverse=True)

        # Determine the number of words to keep (num_words) 
        # and the number of tokens to keep (num_tokens = num_words + 3)
        # according to self.max_num_words and self.minimum_count
		tot_num_words = len(sorted_vocab)
		self.num_words = tot_num_words

		if self.max_num_words:
			self.num_words = min(self.max_num_words, tot_num_words)

		if self.minimum_count:
			frequent = tot_num_words
			for item in reversed(sorted_vocab):
				if item[1] < self.minimum_count:
					frequent -= 1
				else:
					break
			self.num_words = min(self.num_words, frequent)
   
		self.num_tokens = self.num_words + 3  # 3 for the three special tokens: PAD, SOS, EOS
				 
		return sorted_vocab

	def build_word2idx(self, sorted_vocab):
        # Construct a word -> index mapping only for the first self.num_words words plus the 3 tokens
		word_list = []
		word_list.extend(item[0] for item in sorted_vocab[:self.num_words])
		return dict( zip(word_list, list(range(3, self.num_words + 3))) )

	def build_idx2word(self, sorted_vocab):
        # Construct a index -> word mapping only for the first self.num_words words plus the 3 tokens
		word_list = []
		word_list.extend(item[0] for item in sorted_vocab[:self.num_words])
		return dict( zip(list(range(3, self.num_words + 3)), word_list) )

	def trim_corpus(self, tokenized_corpus):
		"""
				Delete the input-output pairs containing words
            that are not included in the first max_num_words 
            words.
		"""
		trimmed_inputs = []
		trimmed_outputs = []
		for input_sentence, output_sentence in zip(tokenized_corpus[0], tokenized_corpus[1]):
			keep_input = True
			keep_output = True
			# Check the input sentence
			for word in input_sentence:
				if word not in self.word2idx:
					keep_input = False
					break
			# Check the output sentence
			if keep_input:
				for word in output_sentence:
					if word not in self.word2idx:
						keep_output = False
						break
			# Only keep pairs that do not contain any word out of the first self.num_words words
			if keep_input and keep_output:
				trimmed_inputs.append(input_sentence)
				trimmed_outputs.append(output_sentence)

		print(f"Deleted {len(tokenized_corpus[0]) - len(trimmed_inputs)} out of {len(tokenized_corpus[0])} pairs, which is {100. * (1.0 - len(trimmed_inputs) / len(tokenized_corpus[0])):.2f}% of the total corpus.")
		return (trimmed_inputs, trimmed_outputs)

	def texts_to_sequences(self, tokenized_corpus, word2idx):
		return list(self.texts_to_sequences_generator(tokenized_corpus, word2idx))

	def texts_to_sequences_generator(self, tokenized_corpus, word2idx):
		for word_seq in tokenized_corpus:
			seq = list()
			for token in word_seq:
				idx = word2idx.get(token)
				if idx is not None:
					seq.append(idx)
				else:
					pass
            # self.max_seq_length represents the maximum length that each sequence
            # can have including the special tokens
			if self.max_seq_length and (len(seq) > self.max_seq_length - 2):
				seq = seq[:self.max_seq_length - 2]
			seq = [1] + seq + [2]
			yield seq

	def pad_minibatch(self, sequences):
		"""
				Find the length of the longest sequence in a
			minibatch (either inputs or outputs, a list of 
            sequences) and pad all sequences to that length.
		"""
		padded_sequences = deepcopy(list(sequences))
		target_len = max([len(x) for x in padded_sequences])
		for index in range(len(padded_sequences)):
			if len(padded_sequences[index]) < target_len:
				if self.front_padded:
					padded_sequences[index] = [0 for i in range(target_len - len(padded_sequences[index]))] + padded_sequences[index]
				else:
					padded_sequences[index].extend([0 for i in range(target_len - len(padded_sequences[index]))])
			else:
				padded_sequences[index] = padded_sequences[index][:target_len]
		return padded_sequences

	def build_embedding_matrix(self, embedding_dim, word_embeddings):
		"""
				Required inputs:
				1. embedding_dim is the embedding dimension
				2. word_embeddings is a dict containing the embeddings
				of words.
		"""
		embedding_matrix = np.zeros((self.num_tokens, embedding_dim))

		for word, idx in self.word2idx.items():
			word_vector = word_embeddings.get(word)
			if word_vector is not None:
				embedding_matrix[idx] = word_vector
			else:
				pass
		
		return embedding_matrix

	def get_sequences_of_test_texts(self, texts):
		"""
				Turn test texts (list of strings) into sequences (list
			of sequence).
		"""
		texts = pd.DataFrame(texts)
		tokenized_texts = self.tokenize_corpus(texts.iloc[:, 0])
		seqs = self.texts_to_sequences(tokenized_texts, self.word2idx)
		return seqs


def corpus_to_vocab(corpus):
	tokenized_input_corpus = [[j for j in i.split() if j] for i in corpus.iloc[:, 0]]
	tokenized_output_corpus = [[j for j in i.split() if j] for i in corpus.iloc[:, 1]]
	vocab = defaultdict(int)
	for text in tokenized_input_corpus:
		for token in text:
			vocab[token] += 1
	for text in tokenized_output_corpus:
		for token in text:
			vocab[token] += 1
	return vocab


class S2SDataset(torch.utils.data.Dataset):
	def __init__(self, X, y):
		super().__init__()
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return (torch.tensor(self.X[idx], dtype = torch.int64), torch.tensor(self.y[idx], dtype = torch.int64))










