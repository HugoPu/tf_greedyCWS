import re
import sys
import collections

import tensorflow as tf

from collections import Counter

from decorator import lazy_property

Py3 = sys.version_info[0] == 3

class CWS_Input:
    def __init__(self, config, train_input=None, name=None):

        self.data_path = config.data_path
        self.is_training = config.is_training

        if not self.is_training:
            self._cache_char_dict = train_input.char_dict
            self._cache_word_dict = train_input.word_dict
        else:
            self.max_sent_len = config.max_sent_len
            self.word_proportion = config.word_proportion
            self.threhold = config.threhold


    @lazy_property
    def lines(self):
        with tf.gfile.GFile(self.data_path, 'r') as f:
            if Py3:
                return f.read().split('\n')
            else:
                return f.read().decode('utf-8').split('\n')

    @lazy_property
    def sentences(self):
        sentences = []
        for line in self.lines:
            sent_left_pointer = 0
            segs = line.split()
            for idx, word in enumerate(segs):
                if len(re.sub('\W', '', word, flags=re.U)) == 0:
                    if idx > sent_left_pointer:
                        sent = ''.join(segs[sent_left_pointer:idx])
                        if not self.is_training or (len(sent) <= self.max_sent_len and len(sent) > 1):
                            sentences.append(segs[sent_left_pointer:idx])
                    sent_left_pointer = idx + 1  # If it is not a punctuation, pointer move right

            # If there isn't any punctuation in the line
            if sent_left_pointer != len(segs):
                sent = ''.join(segs[sent_left_pointer:])
                if not self.is_training or (len(sent) <= self.max_sent_len and len(sent) > 1):
                    sentences.append(segs[sent_left_pointer:])

        return sentences

    @lazy_property
    def char_dict(self):
        character_idx_map = None
        if self.is_training:
            characters = [char for line in self.lines for word in line.split() for char in word]
            counter = collections.Counter(characters)  # Calculate the frequency
            count_pairs = sorted(filter(lambda x: x[1] >= self.threhold, counter.items()),
                                 key=lambda x: -x[1])  # Order words with frequency

            frequent_chars, _ = list(zip(*count_pairs))
            character_idx_map = dict(zip(frequent_chars, range(len(frequent_chars))))
        return character_idx_map

    @lazy_property
    def sents_char_idx(self):
        char_dict = self.char_dict
        sent_char_idx = []
        for sent_words in self.sentences:
            sent = ''.join(sent_words)
            sent_char_idx.append([char_dict[character] if character in char_dict else 0 for character in sent])
        return sent_char_idx

    @lazy_property
    def sents_char_label(self):
        if not self.is_training: return None
        sent_char_label = []
        for sent_words in self.sentences:
            labels = []
            for word in sent_words:
                length = len(word)
                for i in range(length-1):
                    labels.append(0)
                labels.append(length)
            sent_char_label.append(labels)

        return sent_char_label

    @lazy_property
    def word_dict(self):
        # Generate frequent word matrix H
        known_words = None
        if self.word_proportion > 0:
            word_counter = Counter()
            # Loop characters in sentence
            for chars, labels in zip(self.sents_char_idx, self.sents_char_label):
                # Loop labels from 1 to n
                # Generate tuple word index combinations (idx)
                # Count the number of occurrence
                word_counter.update(tuple(chars[idx - label:idx]) for idx, label in enumerate(labels, 1))
            # Put most frequent word in list
            known_word_count = int(self.word_proportion * len(word_counter))
            known_words = dict(word_counter.most_common()[:known_word_count])  # {idx: number of occurrence}
            idx = 0
            # Set know_words to {idx1:idx2}
            for word in known_words:
                known_words[word] = idx
                idx += 1
            # we keep a short list H of the most frequent words, generate parameter matrix H
            # Add known_words and param['word_embed'] as lookup_parameters
        return known_words