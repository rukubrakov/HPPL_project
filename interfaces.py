import numpy as np

def file_to_n_files(filename,n):
    '''Reads .txt file, splits it into n equal parts, saves to the same filepath, but with additional symbols in name, returns nothing.
    For example: file_to_n_files('file.txt',2) will splitt file into 2 with names file_1.txt and file_2.txt'''
    
    #for Linux gsplit -> split
    get_ipython().system('gsplit -d -l $(($(wc -l < "$filename")/$n)) --additional-suffix=.txt $filename $"$(cut -d\'.\' -f1 <<< $filename)_"')
    number_of_words_in_the_last_output = get_ipython().getoutput('(wc -w < $(ls $"$(cut -d\'.\' -f1 <<< $filename)_"* | sort | tail -n 1))')
    if int(number_of_words_in_the_last_output[0]) == 0:
        get_ipython().system('rm $(ls $"$(cut -d\'.\' -f1 <<< $filename)_"* | sort | tail -n 1)    ')



def txt_to_array(filename):
    '''Given filename, for example 'data/file.txt', returns an array of str with all rows in file'''
    
    x = []
    with open(file_name, "r") as f:
        temp_list = []
        for line in f:
            if line.strip(): #line is not blank
                temp_list.append(line[:-2])
            else: #line is blank, i.e., it contains only newlines and/or whitespace
                if temp_list: #check if temp_list contains any items
                    x.append(temp_list)
                temp_list = []
    return x



def array_to_counts(array, n_use, n_keep):
    '''Given the array of str (output of function txt_to_array) first  it counts all the words but in the process it keeps only top n_uses words.
    Then it returns only top n_keep words with their count in the form of 2 arrays, one with strs and another one with ints, ints should be sorted in descending
    order'''
    pass
    #return words, counts

def counts_to_top(words_array, counts_array, n):
    '''Given a lists of words and counts (outputs of array_to_counts for several processes) returns the final top n in the format of 2 arrays,
    one with strs and another one with ints, ints should be sorted in descending order'''
    pass
    # return words, counts

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
    words, counts = array_to_counts(array, n_use, n_keep)
    words_array = comm.gather(words, root=0)
    counts_array = comm.gather(counts, root=0)
    if rank == 0:
        words, counts = counts_to_top(words_array,counts_array,n_final_top)
        np.array(words).dump(folder_to_save_result + 'words' + result_suff + '.txt')
        np.array(counts).dump(folder_to_save_result + 'counts' + result_suff + '.txt')
