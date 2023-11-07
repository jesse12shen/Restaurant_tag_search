import numpy as np
import argparse
import pandas as pd
import csv
import sys
import private_model.text2glove as t2g
"""
text2glove.py requires an import and a private folder out of my caution for
CMU 10-601's retroactive academic policy: as a previous assignment I want
to be on the safer side and refer to a private, modified copy-- only 
available upon request. Functions that I use will be discussed as they come
up.

text2glove.py performs the role of transforming a review into a collection
of GLoVe values and records duplicates while excluding any words outside of
the GLoVe dictionary.
"""
import private_model.sigmoid_train as s_t
"""
sigmoid_train.py requires an import and a private folder out of my caution for
CMU 10-601's retroactive academic policy: as a previous assignment I want
to be on the safer side and refer to a private, modified copy-- only 
available upon request. Functions that I use will be discussed as they come
up.

sigmoid_train.py trains a logistic regression model which I use to classify a
restaraunt review as good or bad.
"""
# import sys
# sys to later be appended in argument for text2glove.py directory
# import os

VECTOR_LEN = 100  # Length of glove vector, from CMU 10-601
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
# in order to work around the error message I get with large csv fields:
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def sigmoid(x: np.ndarray):
    """
    From CMU 10-601 starter code:
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """

    e = np.exp(x)
#     if e == np.inf:
#         return 1.0
#     else:
#         return e / (1 + e)
    return e / (1 + e)
def load_feature_dictionary(file):
    """
    Based on CMU 10-601 starter code:
    Creates a map of words to vectors using the file that has the glove
    embeddings.
    For this version, the word vectors I got from github,
    https://github.com/stanfordnlp/GloVe, are delimited by spaces.


    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        #read_file = csv.reader(f, delimiter='\t')
        read_file = csv.reader(f, delimiter=' ')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

# def filter(weight, data):
#     """
#     Initial good-bad binary classifier. Uses sigmoid(x) function. Breaks ties
#     by picking 1.
#
#     Parameters:
#         data (np.ndarray): the unlabeled data for unsupervised learning
#     Returns:
#             An np.ndarray of binary values equal to data.len
#     """
#
#     answer = np.zeros(len(data))
#     finalanswer = np.zeros(len(data), dtype=int)
#     # prepend a 1 for bias:
#     # one = np.ones(3)
#     new_data = np.hstack((np.ones((len(data), 1)), data))
#     for i in range(len(data)):
#         answer[i] = sigmoid(np.longdouble(np.dot(weight, new_data[i])))
#         if answer[i] >= 0.5:
#             finalanswer[i] = 1
#         else:
#             finalanswer[i] = 0
#
#     return finalanswer, losses
    #return None
def predict_loss(
    theta : np.ndarray,
    X : np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: Implement `predict` using vectorization
    answer = np.zeros(len(X[:, 0]))
    finalanswer = np.zeros(len(X[:, 0]), dtype=int)
    x_i: ndarray = np.insert(X, 0, 1, axis=1) # prepending bias
    for i in range(len(X)):
        answer[i] = sigmoid(np.dot(theta, x_i[i]))
        if answer[i] >= 0.5:
            finalanswer[i] = 1
        else:
            finalanswer[i] = 0
    return finalanswer, answer

if __name__ == '__main__':
    # From CMU 10-601 starter code:
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()

    parser.add_argument("reviews", type=str,
                        help='txt file containing all reviews for clustering')
    parser.add_argument("GLoVe", type=str,
                        help='txt file containing GLoVe embeddings')
    # parser.add_argument("t2g_dir", type=str,
    #                      help='directory location for text2glove.py')
    """
    args.t2g_dir served as a directory location, but after encountering some
    problems, simply importing as part of a package is a simple work-around.
    """
    # TODO: add fleshed out except statement
    try:
         parser.add_argument("goodbad_m", type=str,
                         help='''logistic weights (using GloVe as of 8/2023) for good-bad pass''')
    except:
        print('No weights found, training a new model...')
        s_t.train()
    args = parser.parse_args()
    # try:
    #     import text2glove.py as t2g
    # except:
    #     print('attempting to find text2glove.py in t2g_dir')
    #     # sys.path.append(args.t2g_dir)
    #     import private_model.text2glove as t2g

# w_gb = np.loadtxt(args.goodbad_m, delimiter='\n')
w_gb = np.loadtxt(args.goodbad_m)
#input_data = np.loadtxt(args.reviews, delimiter='\t')
# input_data = np.loadtxt(args.reviews, delimiter='\t', encoding='utf-8',
# dtype='l,O') # I'm going to ignore issues with parentheses and quotes because these are just as valuable
# input_data = pd.read_csv(args.reviews,sep='\t', index_col=False) # seems to cut off 2nd(1) index
input_data = pd.read_csv(args.reviews,sep='\t', usecols=['Text']) #somehow this setting works
# print(str(input_data.iloc[2]))
#input_data = pd.read_csv(args.reviews, usecols=['Text'])
map = load_feature_dictionary(args.GLoVe)
# note that processing data does not involve labels
# reorganize into function for readability
features_gb = np.zeros([len(input_data), VECTOR_LEN])
features_gb_d = features_gb.copy()
words_gb = np.empty(len(input_data), dtype=np.ndarray)
counts_gb = np.empty(len(input_data), dtype=np.ndarray)
for l in range(len(input_data)):
    features_gb[l] = t2g.trim(input_data.iloc[l].Text, map)  # element 0 is
    # the label
for l_temp in range(len(input_data)):
    features_gb[l_temp], words_gb[l_temp], counts_gb[l_temp] = t2g.trim_debug(
        input_data.iloc[l_temp].Text, map)
    '''
    t2g.trim(input: str, GLoVe: dict) is a function that first removes any words 
    not listed in GLoVe and takes a string of many English words and returns 
    the weighted average GLoVe embedding.
    Parameters:
    input (str): a review, a string with many English words
    GLoVe (dict): a dictionary of words to GLoVe embeddings
    Returns:
    Weighted average of GLoVe embeddings for the input string, weighted upon 
    word appearances.
    '''
#features_gb = np.around(features_gb, 6)
#np.savetxt(args.train_out, features_gb, delimiter='\t', newline='\n', fmt='%6f')
#goodbad = pd.DataFrame()
goodbad, likelihood = predict_loss(w_gb, features_gb)
# TODO: improve this model, maybe boosted version?
goodonly = input_data.iloc[([indices for indices in goodbad == 1])]
if goodonly.size == 0:
    raise Exception('Empty "good" restaraunt list')
print('debugging draft')
