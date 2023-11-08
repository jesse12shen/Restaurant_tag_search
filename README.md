# Restaurant_tag_search

I wanted to write a script that takes a collection of restaurant reviews that pass a generous definition of "good" while clustering restaurants with similar traits (e.g. chicken, Mexican, casual etc.). As of this writing, my script will first classify the review as "good" or not using logistic regression, with feature engineered word embeddings, with an intent to skew towards "good", i.e. a bad review has to be exceedingly bad and should be an easy threshold to clear. Next, I want to take the words used and group reviews together to be read in the future if I want to look for a certain kind of restaurant.

Because of my machine learning class's retroactive cheating policy, I cannot post the logistic regression model on a public forum like Github given its similarity to the assignment, but I can share it upon request if appropriate. For this reason, I have a private folder and input weights as an argument to the relevant script. Also for this same reason, I am keeping certain methods of embedding processing hidden, but will provide a description to implement them.
<!-- guess I don't need clustering? Maybe see if I can cluster distinct identities after filtering words-->
## Dependencies
- Original GLoVe embeddings from https://github.com/stanfordnlp/GloVe
I used the pre-trained word vector as my original source: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)
  * The truncated and processed file is ~500 MB so I also needed to use git-lfs and its relevant setup, https://github.com/git-lfs/git-lfs/tree/main


## Areas for Improvement
- Boosting instead of singular logistic regression should increase accuracy-- my plan is adaboosting which I can publicly post.
- Finding a way to extract the article text from webpages-- likely Beautiful Soup. https://beautiful-soup-4.readthedocs.io/en/latest/
- Optimize a GLoVe dictionary with restaurant vocabulary to streamline computation times.
- Look into clustering categories.
- Integrating other NLP methods?
