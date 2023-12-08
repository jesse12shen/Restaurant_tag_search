# Restaurant_tag_search

I wanted to write a script that takes a collection of restaurant reviews that pass a generous definition of "good" while clustering restaurants with similar traits (e.g. chicken, Mexican, casual etc.). As of this writing, my script will first classify the review as "good" or not using logistic regression, with feature engineered word embeddings, with an intent to skew towards "good", i.e. a bad review has to be exceedingly bad and should be an easy threshold to clear. 

After finishing the filtering classifier, I want to take the words used and group reviews together to be read in the future if I want to look for a certain kind of restaurant.

Because of my machine learning class's retroactive cheating policy, I cannot post the logistic regression model on a public forum like Github given its similarity to the assignment, but I can share it upon request if appropriate. For this reason, I have a private folder and input weights as an argument to the relevant script. Also for this same reason, I am keeping certain methods of embedding processing hidden, but will provide a description to implement them.
<!-- guess I don't need clustering? Maybe see if I can cluster distinct identities after filtering words-->
## Dependencies
- Original GLoVe embeddings from https://github.com/stanfordnlp/GloVe
### GLoVe setup
I used the pre-trained word vector as my original source: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)
  * glove_shrink_relation.py has been my preferred way to reduce the size of the original dictionary to a more workable level. The script requires these arguments:
    "feature_dictionary_in", type=str: path to the GloVe feature dictionary .txt file.
    "G_length", type=int: GLoVe length i.e. length of feature vectors.
    "output", type=str: Output txt file name, the smaller dictionary file that will be exported in text form.
    "fold", type=int: Fold of GLoVe division, the output file's size will be dictated by this: total size / fold 
    "central_word", type=str: Word to center GLoVe filtering on. https://nlp.stanford.edu/projects/glove/ describes how nearest neighbors share a high degree of relevancy in the vector space.
    My command line is set to: $ python glove_shrink_relation.py\ glove.6B.300d.txt\ 300\ 300_1e4_glove_embedding.tsv\ 40\ food
  *  When testing the large dictionary: the truncated and processed file can be ~500 MB so I also needed to use git-lfs and its relevant setup, https://github.com/git-lfs/git-lfs/tree/main
  * 300_1e4_glove_embedding.tsv balances runtime and depth fine -- the 1e4  referring to ~10,000 entries.
### Review Filtering Classifier
As of 11/23: the classifier I use is a logistic regression model. Because of its similarity to a class assignment, I have a private sigmoid_train.py module that computes weights and calculates likelihoods. This model is available upon request and takes a GLoVe-calculated dataset of test and training data to produce weights, and reference data, for a given vector length, learning rate, and number of epochs.
### Review data
As of 11/23: reviews.txt consists of 3 excerpts from Oregonlive.com: https://www.oregonlive.com/dining/2023/08/pan-con-queso-is-the-next-portland-pizzeria-you-need-to-know-about.html, https://www.oregonlive.com/dining/2023/08/grana-pdx-specializes-in-folded-neapolitan-pizzas-rick-steves-is-to-thank.html, https://www.oregonlive.com/dining/2023/08/i-tried-to-dine-out-for-a-day-on-only-20-in-eugene-heres-what-happened.html. The fourth entry is a toy example with food terms, an article, and 2 "good"'s. The model itself was trained on a subset of the yelp dataset (https://www.yelp.com/dataset) provided by the assignment. 

## Areas for Improvement
- Boosting instead of singular logistic regression should increase accuracy-- my plan is adaboosting which I can publicly post.
- Finding a way to extract the article text from webpages-- likely Beautiful Soup. https://beautiful-soup-4.readthedocs.io/en/latest/
- Optimize a GLoVe dictionary with restaurant vocabulary to streamline computation times.
- Look into clustering categories.
- Integrating other NLP methods?
