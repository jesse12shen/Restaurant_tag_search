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
#TO DO: incorporating sigmoid_train into the script?
sigmoid_train.py requires an import and a private folder out of my caution for
CMU 10-601's retroactive academic policy: as a previous assignment I want
to be on the safer side and refer to a private, modified copy-- only 
available upon request. Functions that I use will be discussed as they come
up.

sigmoid_train.py trains a logistic regression model which I use to classify a
restaurant review as good or bad.
"""
# import sys
# sys to later be appended in argument for text2glove.py directory
# import os


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
    return e / (1 + e)
def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding dictionary(that will be converted into a np.ndarray).
    """
    allwords = pd.read_csv(file, sep=',', on_bad_lines='warn',
                           engine='python', index_col=0, header=None, quoting=3,
                           escapechar='/')
    glove_map = allwords.to_dict(orient='index')
    return glove_map


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
    parser.add_argument("goodbad_m", type=str,
                        help='''logistic weights (using GloVe as of 8/2023) for good-bad pass''')
    # TODO: add fleshed out except statement
    # try:
    #      parser.add_argument("goodbad_m", type=str,
    #                      help='''logistic weights (using GloVe as of 8/2023) for good-bad pass''')
    # except:
        #print('No weights found, training a new model...')
        #s_t.train()
    parser.add_argument("G_length", type=int,
                        help='GLoVe length i.e. length of feature vectors.')
    args = parser.parse_args()
VECTOR_LEN = args.G_length  # Length of glove vector
w_gb = np.loadtxt(args.goodbad_m)
input_data = pd.read_csv(args.reviews,sep='\t', usecols=['Text'])
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
goodbad, likelihood = s_t.predict_like(w_gb, features_gb)
'''
s_t.predict_loss(weights: np.ndarray, features: np.ndarray) is a function that
calculates likelihood based on the sigmoid probability distribution 
function. This function returns a label vector of ints equal in column 
length to the feature vector, i.e. it is the same count of datapoints, 
and likelihood with those same dimensions.  A copy of the 
sigmoid function used in sigmoid_train.py is 
included in this script. 

Bias does not need to be in features, instead it is
prepended to the features vector in the function itself. 
A likelihood greater than 0.5 is labelled as 1, less is labelled 0.
'''
# TODO: improve this model, maybe boosted version?
goodonly = input_data.iloc[([indices for indices in goodbad == 1])]
if goodonly.size == 0:
    print('Empty "good" restaurant list')

print('work-in-progress version')
