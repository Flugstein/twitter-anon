# Preprocessing of Twitter Data
#
# Author: Florian Lugstein (flugstein@cs.sbg.ac.at)  
# Date: 2020-07-29
#
# Documentation: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/intro-to-tweet-json
# Examples: https://gwu-libraries.github.io/sfm-ui/posts/2016-11-10-twitter-interaction

import sys
import os
import psutil
import json
import time
import numpy as np
import pandas as pd
from pandas import json_normalize
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')


starttime = time.time()

##############################################################################
## Input
##############################################################################

if len(sys.argv) != 5:
    print('usage: python3 twitter_preproc.py tweets.json trans_table_adjlist.txt trans_table_metis.txt out.csv')
    exit()

tweets_path = sys.argv[1]
user_trans_table_adjlist_path = sys.argv[2]
user_trans_table_metis_path = sys.argv[3]
out_path = sys.argv[4]


# Read Dataset JSON File

used_columns = [
    'id_str',
    'truncated',
    'text',
    'entities.hashtags',
    'entities.urls',
    'entities.user_mentions',
    'entities.polls',
    'entities.media',
    'extended_tweet.full_text',
    'extended_tweet.entities.hashtags',
    'extended_tweet.entities.urls',
    'extended_tweet.entities.user_mentions',
    'extended_tweet.entities.polls',
    'extended_tweet.entities.media',
    'retweeted_status.id_str',
    'retweeted_status.text',
    'retweeted_status.entities.hashtags',
    'retweeted_status.entities.urls',
    'retweeted_status.entities.user_mentions',
    'retweeted_status.entities.media',
    'retweeted_status.extended_tweet.full_text',
    'retweeted_status.extended_tweet.entities.hashtags',
    'retweeted_status.extended_tweet.entities.urls',
    'retweeted_status.extended_tweet.entities.user_mentions',
    'retweeted_status.extended_tweet.entities.media',
    'quoted_status.id_str',
    'quoted_status.text',
    'quoted_status.entities.hashtags',
    'quoted_status.entities.urls',
    'quoted_status.entities.user_mentions',
    'quoted_status.entities.media',
    'quoted_status.extended_tweet.full_text',
    'quoted_status.extended_tweet.entities.hashtags',
    'quoted_status.extended_tweet.entities.urls',
    'quoted_status.extended_tweet.entities.user_mentions',
    'quoted_status.extended_tweet.entities.media',
    'in_reply_to_status_id_str',
    'quote_count',
    'retweet_count',
    'created_at',
    'user.id_str',
    'user.followers_count',
    'user.friends_count',
    'user.verified',
    'user.default_profile',
    'user.default_profile_image',
    'user.url',
    'user.listed_count',
    'user.statuses_count',
    'user.favourites_count',
    'user.created_at'
]

chunksize = 100000

with open(tweets_path, 'r', encoding='utf8') as f:
    lines = []
    frames = []
    for i, line in enumerate(f):         
        lines.append(json.loads(line))
        if i != 0 and i % chunksize == 0:
            frame = json_normalize(lines)
            frame = frame[frame.columns.intersection(used_columns)] # only store used columns
            frame = frame.fillna(0) # replace NaNs with zeros
            frames.append(frame)
            lines = []
    frame = json_normalize(lines)
    frame = frame[frame.columns.intersection(used_columns)] # only store used columns
    frame = frame.fillna(0) # replace NaNs with zeros
    frames.append(frame)

df = pd.concat(frames, ignore_index=True, sort=False)
del lines
del frames

print('{} tweets with {} features in dataset, using {} bytes of memory'.format(df.shape[0], df.shape[1], df.memory_usage(index=True).sum()))


# Read Transition Table
user_trans_table_adjlist = {}
with open(user_trans_table_adjlist_path, 'r') as f:
    for line in f:
        split_line = line.split(' ')
        user_trans_table_adjlist[split_line[0]] = np.int64(split_line[1])

print('{} user ids in adjlist transition table'.format(str(len(user_trans_table_adjlist))))

user_trans_table_metis = {}
with open(user_trans_table_metis_path, 'r') as f:
    for line in f:
        split_line = line.split(' ')
        user_trans_table_metis[split_line[0]] = np.int64(split_line[1])

print('{} user ids in metis transition table'.format(str(len(user_trans_table_metis))))


# Create Output Dataframe

tf = pd.DataFrame()



##############################################################################
## Tweet Features
##############################################################################

######################################
### Ids
######################################

tf['id'] = np.int64(df.index + 1) # Ids starting at 1

def id2index(id_):
    return id_ - 1

old_to_new_id = {old_id:np.int64(idx+1) for idx, old_id in df['id_str'].items()}  # dict to convert old to new ids


######################################
### Helper Functions
######################################

# Convert old twitter api id to new id for 'retweet_of' or 'reply_of' column
def conv_id(old_id):
    if old_id == 0:
        return 0  # default value if tweet is not a retweet/quote or reply
    
    no_parent = 0  # default value if parent of retweet/quote or reply tweet is not in dataset
    
    return old_to_new_id.get(old_id, no_parent)


# Get the id of the parent tweet for a retweet
# retweets of quotes have both 'quoted_status' and 'retweeted_status' fields
# Row of df
def retweet_of(row):    
    if row['retweeted_status.id_str'] != 0:
        return conv_id(row['retweeted_status.id_str'])
    return 0


# Get the id of the parent tweet for a quote
# retweets of quotes have both 'quoted_status' and 'retweeted_status' fields
# Row of df
def quote_of(row):
    if row['quoted_status.id_str'] != 0:
        return conv_id(row['quoted_status.id_str'])
    return 0


# Remove source tweets and their retweets, quotes and replies 3 days before the last day of the dataset, to fix retweet statistics, since most retweets occur 3 days after the original tweet
# Iterates the tree from bottom to top, and marks the tweet for deletion
# Returns true if tweet should be removed
# Row of tf
def retweet_fix(row):
    retweet_rm = False
    quote_rm = False
    reply_rm = False
    cutoff_days = pd.offsets.Day(3)

    if row['source'] == True:
        if datetime.date(row['date'] + cutoff_days) > row['last_day']:
            return True
    elif row['retweet_of'] != 0:
        parent_row = tf.loc[id2index(row['retweet_of'])]
        if parent_row['source'] == True:
            if datetime.date(parent_row['date'] + cutoff_days) > row['last_day']:
                return True
        retweet_rm = retweet_fix(parent_row)
    elif row['quote_of'] != 0:
        parent_row = tf.loc[id2index(row['quote_of'])]
        if parent_row['source'] == True:
            if datetime.date(parent_row['date'] + cutoff_days) > row['last_day']:
                return True
        quote_rm =  retweet_fix(parent_row)
    elif row['reply_of'] != 0:
        parent_row = tf.loc[id2index(row['reply_of'])]
        if parent_row['source'] == True:
            if datetime.date(parent_row['date'] + cutoff_days) > row['last_day']:
                return True
        reply_rm = retweet_fix(parent_row)
    return retweet_rm or quote_rm or reply_rm


# Remove retweets, quotes and replies of source tweets outside the dataset
# Iterates the tree from bottom to top, and marks the tweet for deletion if an ancestor is outside the dataset
# Row of tf
def remove_retweets_quotes_replies_outside(row):
    quote_rm = False
    reply_rm = False

    if row['source'] == True:
        return False
    if row['retweet'] == True:
        if row['retweet_of'] == 0:
            return True
    if row['quote'] == True:
        if row['quote_of'] == 0:
            return True
        quote_rm = remove_retweets_quotes_replies_outside(tf.loc[id2index(row['quote_of'])])
    if row['reply'] == True:
        if row['reply_of'] == 0:
            return True
        reply_rm =  remove_retweets_quotes_replies_outside(tf.loc[id2index(row['reply_of'])])
    return quote_rm or reply_rm


# Get retweet count for id using retweet_counts: series of (id, n_retweeted)
def get_retweet_count(id_):
    if id_ not in retweet_counts:
        return 0
    return retweet_counts.loc[id_]


# Get quote count for id using quote_counts: series of (id, n_quoted)
def get_quote_count(id_):
    if id_ not in quote_counts:
        return 0
    return quote_counts.loc[id_]


# Returns the correct entity depending on if the tweet is extended or not
# entity_name: 'text', 'hashtags', 'urls', 'user_mentions', 'polls', 'media'
def get_entity(row, entity_name):   
    if row['extended_tweet.full_text'] != 0:
        if entity_name == 'text':
            return row['extended_tweet.full_text']
        return row['extended_tweet.entities.' + entity_name]
    else:
        if entity_name == 'text':
            return row['text']
        return row['entities.' + entity_name]


# Count hashtags/urls/mentions/media for one tweet
# Row of df
def count_entity(row, entity_name):
    entity = get_entity(row, entity_name)
    
    if entity == 0:
        return 0
    
    return len(entity)

    
# Count media of certain type for one tweet
# Row of df
def count_media(row, type_):
    media = get_entity(row, 'media')
    
    if media == 0:
        return 0
    
    count = 0
    for m in media:
        if m['type'] == type_:
            count += 1
    
    return count


# Count text length of one tweet
# Row of df
def count_text(row):  
    return len(get_entity(row, 'text'))


######################################
### Time/Date Columns
######################################

tf['timestamp'] = pd.to_datetime(df['created_at'])  # timestamp with second accuracy

tf['time'] = tf['timestamp'].dt.time  # only time, regardless of date

tf['hour'] = tf['timestamp'].dt.hour  # only hour, regardless of date

tf['date'] = tf['timestamp'].dt.date  # only date

tf['weekday_enc'] = tf['timestamp'].dt.dayofweek  # weekday encoded: (Monday == 0, Tuesday == 1, ..., Sunday == 6)

tf['weekday'] = tf['timestamp'].dt.day_name()  # weekday as string (Monday, Tuesday, ..., Sunday)

tf['first_day'] = tf['date'][0]  # first day of any tweet in the dataset

tf['last_day'] = tf['date'][len(tf) - 1]  # last day of any tweet in the dataset (without retweet fix)


######################################
### Tweet Types
######################################

# Source
# not retweet, not quote, not reply

tf['source'] = (df['retweeted_status.id_str'] == 0) & (df['quoted_status.id_str'] == 0) & (df['in_reply_to_status_id_str'] == 0)


# Retweet

tf['retweet'] = (df['retweeted_status.id_str'] != 0)

tf['retweet_of'] = df.apply(retweet_of, axis=1)


# Quote

tf['quote'] = (df['quoted_status.id_str'] != 0)

tf['quote_of'] = df.apply(quote_of, axis=1)


# Reply

tf['reply'] = df['in_reply_to_status_id_str'] != 0

tf['reply_of'] = df['in_reply_to_status_id_str'].apply(conv_id)


# Remove tweets and compute statistics

# Remove quotes of source tweets outside the dataset
to_remove_outside = tf.apply(remove_retweets_quotes_replies_outside, axis=1)  # series of (id, to_remove), which tweets need to be removed
rem_count_outside = len(to_remove_outside[to_remove_outside == True])
print('{} retweets, quotes and replies of source tweets outside the dataset removed'.format(rem_count_outside))

# Remove source tweets and their retweets, quotes and replies 3 days before the last day of the dataset, to fix retweet/quote/reply statistics, since most retweets occur within 3 days after the source tweet
to_remove_3_day_fix = tf.apply(retweet_fix, axis=1)
rem_count_3_day_fix = len(to_remove_3_day_fix[to_remove_3_day_fix == True])
print('{} tweets removed for 3 day fix'.format(rem_count_3_day_fix))

# Remove marked tweets
to_remove = to_remove_outside | to_remove_3_day_fix
tf.drop(to_remove[to_remove == True].index, inplace=True)
del to_remove, to_remove_outside, to_remove_3_day_fix, rem_count_outside, rem_count_3_day_fix

# Compute retweet statistics
retweet_counts = tf[tf['retweet_of'] != 0]['retweet_of'].value_counts()  # series of (id, n_retweeted), how many times did tweet get retweeted within the dataset
tf['retweeted_count'] = tf['id'].apply(get_retweet_count)
del retweet_counts

# Compute quote statistics
quote_counts = tf[tf['quote_of'] != 0]['quote_of'].value_counts()  # series of (id, n_quoted), how many times did tweet get quoted within the dataset
tf['quoted_count'] = tf['id'].apply(get_quote_count)
del quote_counts


######################################
### Tweet Content
######################################

# If tweet is not extended, use standard tweet
# If tweet is extended, use extended_tweet
# If tweet is a retweet, use retweeted_status
# If tweet is a retweet and the original tweet is extended, use retweeted_status.extended_tweet

# Hashtags

tf['hashtag_count'] = df.apply(count_entity, args=('hashtags',), axis=1)

tf['hashtag'] = tf['hashtag_count'] != 0


# URLs

tf['url_count'] = df.apply(count_entity, args=('urls',), axis=1)

tf['url'] = tf['url_count'] != 0


# Mentions

tf['mention_count'] = df.apply(count_entity, args=('user_mentions',), axis=1)

tf['mention'] = tf['mention_count'] != 0


# Polls

tf['poll'] = (df['entities.polls'] != 0) | (df['extended_tweet.entities.polls'] != 0)


# Media

tf['media_count'] = df.apply(count_entity, args=('media',), axis=1)

tf['media'] = tf['media_count'] != 0

tf['photo_count'] = df.apply(count_media, args=('photo',), axis=1)

tf['video_count'] = df.apply(count_media, args=('video',), axis=1)

tf['animated_gif_count'] = df.apply(count_media, args=('animated_gif',), axis=1)


# Text length

tf['text_length'] = df.apply(count_text, axis=1)

tf['text_length_median'] = tf['text_length'].median()



##############################################################################
## User Features
##############################################################################

######################################
### Helper Functions
######################################

# Convert old twitter api to new user id using transition table
def conv_user_id_adjlist(id_):   
    not_found_value = 0
    return user_trans_table_adjlist.get(id_, not_found_value)


def conv_user_id_metis(id_):   
    not_found_value = 0
    return user_trans_table_metis.get(id_, not_found_value)


# How active is the user in terms of produced tweets (source tweets and retweets)
# Row of tf
def user_activity(row):
    if row['user.account_age'] == 0:
        return None
    last_tweet_statuses_count = tf.loc[id2index(row['user.last_tweet'])]['user.statuses_count']
    return last_tweet_statuses_count / row['user.account_age']


# How active is the user in terms of produced tweets and likes
# Row of tf
def user_tweets_likes_activity(row):
    if row['user.account_age'] == 0:
        return None
    return (row['user.statuses_count'] + row['user.favourites_count']) / row['user.account_age']


# In which quantile is the property
def quantile(prop_value, q1, q2, q3):
    if prop_value < q1:
        return 0
    elif prop_value < q2:
        return 1
    elif prop_value < q3:
        return 2
    else:
        return 3


######################################
### User ID
######################################

tf['user.adjlist_id'] = df['user.id_str'].apply(conv_user_id_adjlist)
n_not_found_ids = len(tf[tf['user.adjlist_id'] == 0])
if (n_not_found_ids != 0):
    print(str(n_not_found_ids) + ' user ids not found in adjlist transition table')

tf['user.metis_id'] = df['user.id_str'].apply(conv_user_id_metis)
n_not_found_ids = len(tf[tf['user.metis_id'] == 0])
if (n_not_found_ids != 0):
    print(str(n_not_found_ids) + ' user ids not found in metis transition table')



######################################
### Followers
######################################

tf['user.followers_count'] = df['user.followers_count']
q1 = tf['user.followers_count'].quantile(q=0.25)
q2 = tf['user.followers_count'].quantile(q=0.5)
q3 = tf['user.followers_count'].quantile(q=0.75)
tf['user.followers_count_quantile'] = tf['user.followers_count'].apply(quantile, args=(q1, q2, q3))

tf['user.followees_count'] = df['user.friends_count']
q1 = tf['user.followees_count'].quantile(q=0.25)
q2 = tf['user.followees_count'].quantile(q=0.5)
q3 = tf['user.followees_count'].quantile(q=0.75)
tf['user.followees_count_quantile'] = tf['user.followees_count'].apply(quantile, args=(q1, q2, q3))

del q1, q2, q3


######################################
### Profile Info
######################################

tf['user.verified'] = df['user.verified']
tf['user.default_profile'] = df['user.default_profile']
tf['user.default_profile_image'] = df['user.default_profile_image']
tf['user.url'] = df['user.url'] != 0

tf['user.listed_count'] = df['user.listed_count']
tf['user.listed_count_mean'] = tf['user.listed_count'].mean()


######################################
### Activity
######################################

lutf = tf[['user.metis_id', 'id']].drop_duplicates(subset='user.metis_id', keep='last').set_index('user.metis_id', drop=True)  # table of last tweet of every user
tf['user.last_tweet'] = tf['user.metis_id'].apply(lambda user_id: lutf.loc[user_id]['id'])
del lutf

tf['user.statuses_count'] = df['user.statuses_count']

tf['user.favourites_count'] = df['user.favourites_count']

tf['user.account_age'] = (tf['last_day'] - pd.to_datetime(df['user.created_at']).dt.date).dt.days
tf['user.account_age'] = tf['user.account_age'].astype(int)

tf['user.activity'] = tf.apply(user_activity, axis=1)

tf['user.mean_activity'] = tf.drop_duplicates(subset='user.metis_id', keep='last')['user.activity'].mean()

tf['user.tweets_likes_activity'] = tf.apply(user_tweets_likes_activity, axis=1)



##############################################################################
## Seperate Text Data
##############################################################################

txtf = pd.DataFrame()  # seperate text dataframe

out_path_split = os.path.splitext(out_path)
txt_out_path = out_path_split[0] + '_tweet_text' + out_path_split[1]

tknzr = TweetTokenizer(preserve_case=False)
stops = Counter(stopwords.words('german') + stopwords.words('english'))

def anon_text(row):
    text = get_entity(row, 'text')
    
    # anon mentions and replace shortened urls with expanded versions
    mentions_and_urls = get_entity(row, 'user_mentions') + get_entity(row, 'urls')
    mentions_and_urls.sort(key=lambda e: e['indices'][0])
    
    offset = 0
    for mu in mentions_and_urls:
        if 'screen_name' in mu:
            # mention
            mention = mu
            user_id = str(conv_user_id_adjlist(mention['id_str']))
            indices = mention['indices']

            text = text[:indices[0]+offset] + '@user' + user_id + text[indices[1]+offset:]
            offset += len('@user' + user_id) - (indices[1] - indices[0])
        else:
            # url
            url = mu
            expanded_url = url['expanded_url']
            indices = url['indices']

            text = text[:indices[0]+offset] + expanded_url + text[indices[1]+offset:]
            offset += len(expanded_url) - (indices[1] - indices[0])

    # remove stop words
    text = [word for word in tknzr.tokenize(text) if word not in stops]
    
    # remove link to tweet, which is present if it contains media
    if count_entity(row, 'media') > 0:
        text = text[:-1]
    
    return text


txtf['id'] = tf['id']

txtf['text'] = df.apply(anon_text, axis=1)

txtf['text_sorted'] = txtf['text'].apply(sorted)



##############################################################################
## Save dataframe as CSV file
##############################################################################

process = psutil.Process(os.getpid())
print('Process using {} bytes of memory'.format(process.memory_info().rss))

tf.to_csv(out_path, index=False)
txtf.to_csv(txt_out_path, index=False)

print('Runtime: %is' % (time.time() - starttime))
