# Extracting Simulation Features from Preprocessed Twitter Data
#
# Author: Florian Lugstein (flugstein@cs.sbg.ac.at)  
# Date: 2021-07-29

import sys
import numpy as np
import pandas as pd


##############################################################################
## Input
##############################################################################

if len(sys.argv) != 3:
    print('usage: python3 sim_features.py tweets_preproc.csv out.csv')
    exit()

preproc_path = sys.argv[1]
out_path = sys.argv[2]

tf = pd.read_csv(preproc_path)

# Create Output Dataframe
sf = pd.DataFrame()


##############################################################################
## Simulation Features
##############################################################################

# Helper Functions

def binary_bool(value):
    return 1 if value else 0

def get_retweeters_adjlist(id_):
    return list(tf[tf['retweet_of'] == id_]['user.adjlist_id'])

def get_retweeters_metis(id_):
    return list(tf[tf['retweet_of'] == id_]['user.metis_id'])


# Filter tweets

print('{} source tweets without author removed'.format(tf[((tf['user.adjlist_id'] == 0) | (tf['user.metis_id'] == 0)) & (tf['source'] == True)].shape[0]))
tf = tf[tf['user.adjlist_id'] != 0]  # only tweets with author
tf = tf[tf['user.metis_id'] != 0]  # only tweets with author
tfs = tf[tf['source'] == True]  # only source tweets


# Features

sf['author_adjlist'] = tfs['user.adjlist_id']

sf['author_metis'] = tfs['user.metis_id']

sf['verified'] = tfs['user.verified'].apply(binary_bool)

sf['activity'] = (tfs['user.activity'] >= tfs['user.mean_activity']).apply(binary_bool)

sf['defaultprofile'] = tfs['user.default_profile'].apply(binary_bool)

sf['userurl'] = tfs['user.url'].apply(binary_bool)

sf['hashtag'] = tfs['hashtag'].apply(binary_bool)

sf['tweeturl'] = tfs['url'].apply(binary_bool)

sf['mentions'] = tfs['mention'].apply(binary_bool)

sf['media'] = tfs['media'].apply(binary_bool)

sf['retweeter_adjlist'] = tfs['id'].apply(get_retweeters_adjlist)

sf['retweeter_metis'] = tfs['id'].apply(get_retweeters_metis)


##############################################################################
## Save Simulation Features
##############################################################################

sf.to_csv(out_path, index=False)
