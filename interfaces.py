import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from numba import njit
import numba

def file_to_n_files(filename,n):
    '''Reads .txt file, splits it into n equal parts, saves to the same filepath, but with additional symbols in name, returns nothing.
    For example: file_to_n_files('file.txt',2) will splitt file into 2 with names file_1.txt and file_2.txt'''
    
    #for Linux gsplit -> split
    get_ipython().system('split -d -l $(($(wc -l < "$filename")/$n)) --additional-suffix=.txt $filename $"$(cut -d\'.\' -f1 <<< $filename)_"')
    number_of_words_in_the_last_output = get_ipython().getoutput('(wc -w < $(ls $"$(cut -d\'.\' -f1 <<< $filename)_"* | sort | tail -n 1))')
    if int(number_of_words_in_the_last_output[0]) == 0:
        get_ipython().system('rm $(ls $"$(cut -d\'.\' -f1 <<< $filename)_"* | sort | tail -n 1)    ')

def put_str_together(array, n = 10):
    if len(array) < n * 2:
        return array
    else:
        tmp = []
        for i in range(len(array)//2):
            tmp.append(array[2*i]+' '+array[2*i+1])
        if len(array)%2 != 0:
            tmp.append(array[-1])
        return put_str_together(tmp, n)

    
def txt_to_array(filename):
    '''Given filename, for example 'data/file.txt', returns an array of str with all rows in file'''
    
    x = []
    with open(filename, "r") as f:
        temp_list = []
        for line in f:
            if line.strip(): #line is not blank
                temp_list.append(line.replace('\n',''))
            else: #line is blank, i.e., it contains only newlines and/or whitespace
                if temp_list: #check if temp_list contains any items
                    x += temp_list
                temp_list = []
    return x


def preprocess(document, lemmatization=False, rm_stop_words=False):
    """
    - convert the whole text to the lowercase;
    - tokenize the text;
    - remove stopwords;
    - lemmatize the text.
    Return: string, resulted list of tokens joined with the space.
    """

    # If we want to delete stopwords and perform lemmatization
    # nltk.download('wordnet')
    # nltk.download('stopwords')
    # wordnet_lemmatizer = WordNetLemmatizer()
    # stop_words = set(stopwords.words('english))
    tokenizer = RegexpTokenizer(r'[a-z]+')

    # Convert to lowercase
    document = document.lower()
    # Tokenize
    words = tokenizer.tokenize(document)

    # Removing stopwords
    if rm_stop_words:
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    # Lemmatizing
    if lemmatization:
        #nltk.download('wordnet')
        wordnet_lemmatizer = WordNetLemmatizer()
        for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            words = [wordnet_lemmatizer.lemmatize(x, pos) for x in words]

    return words
def array_to_counts2(array, n_use, n_keep):
    dic = {}
    for i,str in enumerate(array):
        str = preprocess(str, lemmatization=True, rm_stop_words=True)
        for word in str:
            if word in dic:
                dic[word]+=1
            else:
                dic[word] = 1
        if (i+1) % 5 == 0:
            dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])[-n_use:]}
    dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    return list(dic.keys())[:n_keep], list(dic.values())[:n_keep]


def array_to_counts(array, n_use, n_keep):
    '''Given the array of str (output of function txt_to_array) first  it counts all the words but in the process it keeps only top n_uses words.
    Then it returns only top n_keep words with their count in the form of 2 arrays, one with strs and another one with ints, ints should be sorted in descending
    order'''
    tmp_voc = {}
    tmp_array = array.copy()
    # preprocess strings, have the array of sorted dicts of token frequencies
    tmp_array = [dict(sorted(dict(Counter(preprocess(tmp, lemmatization=True, rm_stop_words=True))).items(), key=lambda item: item[1], reverse=True)) for tmp in tmp_array]
    # count how many most frequent word we should take from each str to dill the array of size `n_use`
    words_from_str = n_use // len(tmp_array) + 1 if n_use > len(tmp_array) else n_use
    second_run = 0
    for k, tokens in enumerate(tmp_array):
        i = 0
        keys_to_pop = []
        keys = list(tokens.keys())
        values = list(tokens.values())
        if len(tmp_voc) < n_use + 1:
            # here we add only `m` words from the str to count only most frequent words
            m = min(words_from_str, len(tokens))
            while i < m:
                if keys[i] in tmp_voc:
                    tmp_voc[keys[i]] += values[i]
                    keys_to_pop.append(i)
                    i += 1
                    m = min(len(tokens), m + 1)
                else:
                    if len(tmp_voc) < n_use + 1:
                        tmp_voc[keys[i]] = values[i]
                        keys_to_pop.append(i)
                        i += 1
                    else:
                        break
        else:
            # when we go out of size, just count words from `tmp_voc'
            second_run = k if second_run == 0 else second_run
            for key, value in zip(keys, values):
                if key in tmp_voc:
                    tmp_voc[key] += value
                else:
                    continue
        for idx in keys_to_pop:
            tmp_array[k].pop(keys[idx])
    # also, we need to count words we could miss in the first run
    for k in range(second_run):
        keys = list(tmp_array[k].keys())
        values = list(tmp_array[k].values())
        for key, value in zip(keys, values):
            if key in tmp_voc:
                tmp_voc[key] += value
            else:
                continue
    tmp_voc = dict(sorted(tmp_voc.items(), key=lambda item: item[1], reverse=True))
    return list(tmp_voc.keys())[:n_keep], list(tmp_voc.values())[:n_keep]


def counts_to_top(words_array, counts_array, n):
    '''Given a lists of words and counts (outputs of array_to_counts for several processes) returns the final top n in the format of 2 arrays,
    one with strs and another one with ints, ints should be sorted in descending order'''
    # words_array is a list of m lists, len of each <= n, where m - number of processes
    # return words, counts
    flat_words = [item for sublist in words_array for item in sublist]
    flat_counts = [item for sublist in counts_array for item in sublist]
    dic = dict(zip(set(flat_words), np.zeros(len(set(flat_words)), dtype = int)))
    for i, word in enumerate(flat_words):
        dic[word] += flat_counts[i]
    dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse = True))
    (keys,values) = zip(*dic.items())    
    return list(keys[:min(n, len(keys))]), list(values[:min(n, len(keys))])

def counts_to_top2(words_array, counts_array, n):
    dic = {}
    for word_array,count_array in zip(words_array,counts_array):
        for word,count in zip(word_array,count_array):
            if word in dic:
                dic[word] += count
            else:
                dic[word] = count
    dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse = True))
    (keys,values) = zip(*dic.items())    
    return list(keys[:min(n, len(keys))]), list(values[:min(n, len(keys))])

def accuracy(words_real, counts_real, words_predicted, counts_predicted):
    '''Some metric to estimate the quality of prediction (should be 1 when same argument given for real and predicted
    and 0 if there is no words coincide'''
    hit_rate = len(set(words_real) & set(words_predicted)) / len(set(words_real) | set(words_predicted))

    score = 0
    for i, word in enumerate(words_real):
        if word in words_predicted:
            cp = counts_predicted[[e for e in range(len(words_predicted)) if words_predicted[e] == word][0]]
            cr = counts_real[i]
            score += abs(cp - cr)
        else:
            score += counts_real[i]
    score /= sum(counts_real)
    score = 1 - score

    return hit_rate, score


def pipeline(filename, n_use, n_keep, rank, comm,n_final_top, folder_to_save_result = 'result/',result_suff = '_0'):
    array = txt_to_array(filename)
    words, counts = array_to_counts2(array, n_use, n_keep)
    words_array = comm.gather(words, root=0)
    counts_array = comm.gather(counts, root=0)
    if rank == 0:
        words, counts = counts_to_top2(words_array,counts_array,n_final_top)
        np.array(words).dump(folder_to_save_result + 'words' + result_suff + '.txt')
        np.array(counts).dump(folder_to_save_result + 'counts' + result_suff + '.txt')
