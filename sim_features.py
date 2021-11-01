# Extracting Simulation Features from Preprocessed Twitter Data
#
# Author: Florian Lugstein (flugstein@cs.sbg.ac.at)  
# Date: 2021-10-28

import sys
import numpy as np
import pandas as pd


##############################################################################
## Input
##############################################################################

if len(sys.argv) != 4:
    print('usage: python3 sim_features.py tweets_preproc.csv out.csv mode')
    exit()

preproc_path = sys.argv[1]
out_path = sys.argv[2]
mode = sys.argv[3]

tf = pd.read_csv(preproc_path)

# Create Output Dataframe
sf = pd.DataFrame()


##############################################################################
## Simulation Features
##############################################################################

# Helper Functions

def binary_bool(value):
    return 1 if value else 0

def get_retweeters_hpda(id_):
    return list(tf[tf['retweet_of'] == id_]['user.id'])

def get_retweeters_sim(id_):
    return len(list(tf[tf['retweet_of'] == id_]['user.id']))


# Filter tweets

print('{} source tweets without author removed'.format(tf[((tf['user.id'] == 0) | (tf['user.id'] == 0)) & (tf['source'] == True)].shape[0]))
tf = tf[tf['user.id'] != 0]  # only tweets with author
tfs = tf[tf['source'] == True]  # only source tweets


# Features

sf['author'] = tfs['user.id']

sf['verified'] = tfs['user.verified'].apply(binary_bool)

sf['activity'] = (tfs['user.activity'] >= tfs['user.mean_activity']).apply(binary_bool)

sf['defaultprofile'] = tfs['user.default_profile'].apply(binary_bool)

sf['userurl'] = tfs['user.url'].apply(binary_bool)

sf['hashtag'] = tfs['hashtag'].apply(binary_bool)

sf['tweeturl'] = tfs['url'].apply(binary_bool)

sf['mentions'] = tfs['mention'].apply(binary_bool)

sf['media'] = tfs['media'].apply(binary_bool)

if mode == 'sim':
    sf['retweeters'] = tfs['id'].apply(get_retweeters_sim)
elif mode == 'hpda':
    sf['retweeters'] = tfs['id'].apply(get_retweeters_hpda)
else:
    print("error: wrong mode")
    exit()


##############################################################################
## Save Simulation Features
##############################################################################

sf.to_csv(out_path, index=False)
