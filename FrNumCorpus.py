import numpy as np
import json
from num2words import num2words
import utils

fr_vocab = ["zéro", "virgule", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "vingt","trente", "quarante", "cinquante", "soixante", "cent", "mille", "million", "milliard", "-", "et"]
np.random.seed(1)
def createFrNumCorpus(filename, n):
    num_digit = []
    num_fr = []
    dict_corpus = {}
    for num in np.random.randint(0, 100000, n):
        num_digit.append(str(num))
        num_fr.append(num2words(num, lang='fr'))
        
        num_digit.append(str(0.1*num))
        num_fr.append(num2words(0.1*num, lang='fr'))
        
        num_digit.append(str(0.05*num))
        num_fr.append(num2words(0.05*num, lang='fr'))

    for num in np.random.randint(100000, 1000000000, n):
        num_digit.append(str(num))
        num_fr.append(num2words(num, lang='fr'))
        
        num_digit.append(str(0.1*num))
        num_fr.append(num2words(0.1*num, lang='fr'))
        
        num_digit.append(str(0.05*num))
        num_fr.append(num2words(0.05*num, lang='fr'))
        
    dict_corpus["num_digit"] = num_digit
    dict_corpus["num_fr"] = num_fr
    s = json.dumps(dict_corpus,ensure_ascii=False)

    with open(filename, 'w', encoding='utf8') as json_file:
        json_file.write(s)

    print("Corpus file " + filename + " has been created")


class frNumData:
    def __init__(self, corpus_file, debug=False):
        np.random.seed(1)
        self.num_digit = []
        self.num_fr = []
        with open(corpus_file, 'r', encoding='utf8') as json_file:
            data = json.load(json_file)
        self.num_digit = data["num_digit"]
        self.num_fr = data["num_fr"]

        self.vocab = ["<PAD>"] + [str(i) for i in range(0, 10)]  + list(fr_vocab) + [".", "<BEGIN>", "<END>"]

        self.dict_word_idx = {v: i for i, v in enumerate(self.vocab, start=0)}
        self.dict_idx_word = {i: v for v, i in self.dict_word_idx.items()}

        assert self.dict_word_idx["<PAD>"] == utils.PAD_IDX
        assert self.dict_idx_word[utils.PAD_IDX] == "<PAD>"

        self.x, self.y = [], []
        for digit_sen, fr_sen in zip(self.num_digit, self.num_fr):
            fr_sen = fr_sen.replace("-", " - ")
            fr_sen = fr_sen.replace("cents", "cent")
            fr_sen = fr_sen.replace("vingts", "vingt")
            fr_sen = fr_sen.replace("millions", "million")
            fr_sen = fr_sen.replace("milliards", "milliard")

            self.x.append([self.dict_word_idx[v] for v in digit_sen])
            self.y.append(
                [self.dict_word_idx["<BEGIN>"], ] + [self.dict_word_idx[v] for v in fr_sen.split(" ")] + [
                    self.dict_word_idx["<END>"], ])

        if debug:
            print("Load ", len(self.num_digit), " digital sentences")
            print("Load ", len(self.num_fr), " french sentences")
            print(self.num_digit[0], ": ", self.num_fr[0])
            print("self.vocab: ",self.vocab)
            print("self.dict_word_idx: ",self.dict_word_idx)
            print("self.dict_idx_word: ",self.dict_idx_word)
            print(self.x[0])
            print(self.y[0])
        self.x_maxLen = 0
        self.y_maxLen = 0
        for x in self.x:
            if len(x) > self.x_maxLen:
                self.x_maxLen = len(x)
        print("x_maxLen", self.x_maxLen)
        
        for y in self.y:
            if len(y) > self.y_maxLen:
                self.y_maxLen = len(y)
        print("y_maxLen", self.y_maxLen)
        #self.x, self.y = np.array(self.x), np.array(self.y)
        self.start_token = self.dict_word_idx["<BEGIN>"]
        self.end_token = self.dict_word_idx["<END>"]

    def sample(self, n=64, showSensence=False):
        bi = np.random.randint(0, len(self.x), size=n)
        bx = []
        by = []
        for idx in bi:
            x_item, y_item = self.x[idx], self.y[idx]
            if (showSensence):
                print("ID = ", idx, "  ", self.num_digit[idx], ": ", self.num_fr[idx])
            bx.append(np.array(x_item))
            by.append(np.array(y_item))

        return bx, by

    def fresh_sample(self, showSensence=False, num_input = None):
        fresh_num_digit = []
        fresh_num_fr = []
        if num_input:
            fresh_num_digit.append(str(num_input))
            fresh_num_fr.append(num2words(num_input, lang='fr'))
        else:
            for num in np.random.randint(0, 1000000000, 1):
                fresh_num_digit.append(str(num))
                fresh_num_fr.append(num2words(num, lang='fr'))
        
                fresh_num_digit.append(str(0.01*num))
                fresh_num_fr.append(num2words(0.1*num, lang='fr'))
        
                fresh_num_digit.append(str(0.005*num))
                fresh_num_fr.append(num2words(0.05*num, lang='fr'))
        
        fresh_x, fresh_y = [], []
        for digit_sen, fr_sen in zip(fresh_num_digit, fresh_num_fr):
            # normalize some special formats
            # "-" is treated as a word, so wrap it with delimiter
            fr_sen = fr_sen.replace("-", " - ")
            # Plural form is considered to be the same word in single form, remove "s"
            fr_sen = fr_sen.replace("vingts", "vingt")
            fr_sen = fr_sen.replace("cents", "cent")
            fr_sen = fr_sen.replace("millions", "million")
            fr_sen = fr_sen.replace("milliards", "milliard")

            fresh_x.append([self.dict_word_idx[v] for v in digit_sen])
            fresh_y.append(
                [self.dict_word_idx["<BEGIN>"], ] + [self.dict_word_idx[v] for v in fr_sen.split(" ")] + [
                    self.dict_word_idx["<END>"], ])
        
        bx = []
        by = []
        for idx in range(0, len(fresh_num_fr)):
            x_item, y_item = fresh_x[idx], fresh_y[idx]
            if (showSensence):
                print("ID = ", idx, "  ", fresh_num_digit[idx], ": ", fresh_num_fr[idx])
                print("fresh_x", fresh_x)
                print("fresh_y", fresh_y)
            bx.append(np.array(x_item))
            by.append(np.array(y_item))

        return bx, by
        
    def idx2str(self, idx):
        x = []
        for i in idx:
            x.append(self.dict_idx_word[i])
            if i == self.end_token:
                break
        return " ".join(x)

    @property
    def num_word(self):
        return len(self.vocab)


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--create', type=int, dest='creation', default=None, help='create corpus!')
    
    parser.add_argument('--show', type=str, dest='filename', default=None, help='show corpus!')

    args = parser.parse_args()
    print(args)

    if args.creation:
        n = args.creation
        np.random.seed(n)
        createFrNumCorpus("frNumData_{}.json".format(6*n), n)
    elif args.filename:
        ## show info of the given corpus dataset
        data = frNumData(args.filename, True)    
        data.sample(2, True)
        data.fresh_sample(True)
    else:
        ## show a sample corpus dataset
        data = frNumData("corpus/frNumData_30.json", True)    
        data.sample(2, True)