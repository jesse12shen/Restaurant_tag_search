# Restaurant_tag_search

I wanted to write a script that takes a collection of restaurant reviews that pass a generous definition of "good" while clustering restaraunts with similar traits (e.g. chicken, Mexican, casual etc.). As of this writing, my script will first classify the review as "good" or not using a logistic regression, with feature engineered word embeddings, with an intent to skew towards "good", i.e. a bad review has to be exceedingly bad and should be an easy threshold to clear. Next, I want to take the words used and organize 
