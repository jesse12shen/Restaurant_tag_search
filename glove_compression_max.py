import csv
import numpy as np
import sys
import argparse

import pandas as pd

# VECTOR_LEN = 300  # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

"""
    Script for reducing the size of a .txt GLoVE dictionary. In this version, 
    the top half of vectors with greatest magnitude were saved as a new file.
"""
################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


# maxInt = sys.maxsize
# csv.field_size_limit(maxInt)
def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    map = pd.read_csv(file, sep=' ', on_bad_lines='warn',
                      engine='python', index_col=0, header=None, quoting=3
                      )
    #      quoting=3 solves line reading issues, seems a little faster than
    #      chunking.
    words = map.index
    for k in range(len(words)):
        glove_map[words[k]] = np.array(map.loc[words[k]], dtype=float)
    return glove_map


if __name__ == '__main__':
    # From original copy:
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dictionary_in", type=str,
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("G_length", type=int,
                        help='GLoVe length i.e. length of feature vectors')
    parser.add_argument("output", type=str,
                        help='output txt file name')
    parser.add_argument("draft_yn", type=bool,
                        help='smaller draft version path true/false')
    parser.add_argument("draft_name", type=str,
                        help='smaller draft version to load')
    args = parser.parse_args()
    VECTOR_LEN = args.G_length  # Length of glove vector
    allwords = pd.read_csv(args.feature_dictionary_in, sep=' ',
                           on_bad_lines='warn',
                           engine='python', index_col=0, header=None, quoting=3
                           )
    magnitude = np.sum(np.square(allwords), axis=1)  # expect a series at this
    # point
    magnitude.sort_values()
    ranked_list = magnitude.index.array.copy()
    top_half = allwords.loc[ranked_list[:int(len(ranked_list) / 2)],
               :]  # taking
    # the top half of the ranked list to then be inputted into the
    # dictionary, order doesn't matter at this point.
    np.savetxt(args.output, top_half, delimiter='\t', newline='\n', fmt='%6f')
