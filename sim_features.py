# Extracting Simulation Features from Preprocessed Twitter Data
#
# Author: Florian Lugstein (flugstein@cs.sbg.ac.at)  
# Date: 2020-07-29

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


# Filter tweets

print('{} source tweets without author removed'.format(tf[((tf['user.adjlist_id'] == 0) | (tf['user.metis_id'] == 0)) & (tf['source'] == True)].shape[0]))
tf = tf[tf['user.adjlist_id'] != 0]  # only tweets with author
tf = tf[tf['user.metis_id'] != 0]  # only tweets with author
tf = tf[tf['source'] == True]  # only source tweets


# Features

sf['author_adjlist'] = tf['user.adjlist_id']

sf['author_metis'] = tf['user.metis_id']

sf['verified'] = tf['user.verified'].apply(binary_bool)

sf['activity'] = (tf['user.activity'] >= tf['user.mean_activity']).apply(binary_bool)

sf['defaultprofile'] = tf['user.default_profile'].apply(binary_bool)

sf['userurl'] = tf['user.url'].apply(binary_bool)

sf['hashtag'] = tf['hashtag'].apply(binary_bool)

sf['tweeturl'] = tf['url'].apply(binary_bool)

sf['mentions'] = tf['mention'].apply(binary_bool)

sf['media'] = tf['media'].apply(binary_bool)

sf['retweets'] = tf['retweeted_count']


##############################################################################
## Save Simulation Features
##############################################################################

sf.to_csv(out_path, index=False)
