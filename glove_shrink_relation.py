import csv
import numpy as np
import sys
import argparse

import pandas as pd

MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

"""
    Script for reducing the size of a .txt GLoVE dictionary. In this version, 
    the top half of vectors with greatest magnitude were saved as a new file.
"""
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)




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
                        help='path to the GloVe feature dictionary .txt file.')
    parser.add_argument("G_length", type=int,
                        help='GLoVe length i.e. length of feature vectors.')
    parser.add_argument("output", type=str,
                        help='output txt file name.')
    parser.add_argument("fold", type=int,
                        help='fold of GLoVe division.')
    parser.add_argument("central_word", type=str,
                        help='Word to center GLoVe filtering on.')
    args = parser.parse_args()
    VECTOR_LEN = args.G_length  # Length of glove vector
    allwords = pd.read_csv(args.feature_dictionary_in, sep=' ',
                           on_bad_lines='warn',
                           engine='python', index_col=0, header=None, quoting=3
                           )
    if args.central_word not in allwords.index:
        raise IndexError('Inputted word was not in dataframe index.')
    # duplicate removal for easier dictionary organization:
    allwords = allwords.drop_duplicates(keep='first')
    # Attempting to truncate the dictionary by taking words closest in
    # relation to a central word
    food_rel = np.sum((allwords - allwords.loc[args.central_word])**2, axis=1)
    ranked_list = food_rel.sort_values(axis=0)
    top_portion = allwords.loc[ranked_list[:int(len(ranked_list.index.array) /
                                             args.fold)].index.array,
               :]  # taking
    # the top portion of the ranked list to then be inputted into the
    # dictionary.
    top_portion.to_csv(path_or_buf=args.output, header=False, quoting=3,
                    escapechar='/')