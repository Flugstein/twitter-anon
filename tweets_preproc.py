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
import numpy as np
import pandas as pd
from pandas import json_normalize


##############################################################################
## Input
##############################################################################

if len(sys.argv) != 4:
    print('usage: python3 twitter_preproc.py tweets.json trans_table.txt out.csv')
    exit()

tweets_path = sys.argv[1]
user_trans_table_path = sys.argv[2]
out_path = sys.argv[3]


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
    'in_reply_to_status_id_str',
    'retweet_count',
    'quoted_status.id_str',
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
user_trans_table = {}
with open(user_trans_table_path, 'r') as f:
    for line in f:
        split_line = line.split(' ')
        user_trans_table[split_line[0]] = np.int64(split_line[1])

print('{} user ids in transition table'.format(str(len(user_trans_table))))


# Create Output Dataframe

tf = pd.DataFrame()

process = psutil.Process(os.getpid())
print('Process using {} bytes of memory'.format(process.memory_info().rss))



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

# Convert old id to new id for 'reply_of' or 'retweet_of' or 'quoted_of' column
def conv_id(old_id):
    if old_id == 0:
        return 0  # default value if tweet is not a reply/retweet/quote
    
    no_parent = 0  # default value if parent of reply/retweet/quote tweet is not in dataset
    
    return old_to_new_id.get(old_id, no_parent)


# Remove source tweets and their retweets x days before the last day of the dataset, to fix retweet statistics, since most retweets occur x days after the original tweet
# Returns true if tweet should be removed
def retweet_fix(row):
    cutoff_days = pd.offsets.Day(3)
    if row['pure_source'] == True:
        if row['date'] + cutoff_days > row['last_day']:
            return True
    elif row['retweet_of'] != 0:
        if tf.loc[id2index(row['retweet_of'])]['date'] + cutoff_days > row['last_day']:
            return True
    return False


# Get retweet count for id using retweet_counts: series of (id, n_retweeted)
def get_retweet_count(id_, retweet_counts):
    if id_ not in retweet_counts:
        return 0
    return retweet_counts.loc[id_]


# Count hashtags/urls/mentions/media for one tweet
def count_entity(row, entity):
    if row['retweeted_status.id_str'] == 0:
        if row['truncated']:
            column = 'extended_tweet.entities.' + entity
        else:
            column = 'entities.' + entity
    else:
        if row['retweeted_status.extended_tweet.full_text'] != 0:
            column = 'retweeted_status.extended_tweet.entities.' + entity
        else:
            column = 'retweeted_status.entities.' + entity
        
    if row[column] == 0:
        return 0
    
    return len(row[column])

    
# Count media of certain type for one tweet
def count_media(row, type_):
    if row['retweeted_status.id_str'] == 0:
        if row['truncated']:
            column = 'extended_tweet.entities.media'
        else:
            column = 'entities.media'
    else:
        if row['retweeted_status.extended_tweet.full_text'] != 0:
            column = 'retweeted_status.extended_tweet.entities.media'
        else:
            column = 'retweeted_status.entities.media'
    
    if row[column] == 0:
        return 0
    
    count = 0
    for media in row[column]:
        if media['type'] == type_:
            count += 1
    
    return count


# Count length of text for one tweet
def count_text(row):
    if row['retweeted_status.id_str'] == 0:
        if row['truncated']:
            column = 'extended_tweet.full_text'
        else:
            column = 'text'
    else:
        if row['retweeted_status.extended_tweet.full_text'] != 0:
            column = 'retweeted_status.extended_tweet.full_text'
        else:
            column = 'retweeted_status.text'
            
    return len(row[column])


######################################
### Time/Date Columns
######################################

tf['date'] = pd.to_datetime(df['created_at'])  # whole date

tf['time'] = tf['date'].dt.time  # only time, regardless of day

tf['hour'] = tf['date'].dt.hour  # only hour, regardless of day

tf['day'] = tf['date'].dt.day  # only day

tf['weekday_enc'] = tf['date'].dt.dayofweek  # weekday encoded: (Monday == 0, Tuesday == 1, ..., Sunday == 6)

tf['weekday'] = tf['date'].dt.day_name()  # weekday as string (Monday, Tuesday, ..., Sunday)

tf['first_day'] = tf['date'][0]  # first day of any tweet in the dataset

tf['last_day'] = tf['date'][len(tf) - 1]  # last day of any tweet in the dataset


######################################
### Tweet Types
######################################

# Pure source
# not reply, not retweet

tf['pure_source'] = (df['retweeted_status.id_str'] == 0) & (df['in_reply_to_status_id_str'] == 0)


# Retweet
# 'retweet'==True and 'retweet_of'==0 => tweet is a retweet, but parent is not in dataset
# 'retweeted_count_int' => number of retweets of this tweet in the dataset
# 'retweeted_count_ext' => number of retweets of this tweet in and outside the dataset
# 'retweeted'==True => there is a retweet of this tweet in the dataset (retweets of retweets also count)

tf['retweet'] = df['retweeted_status.id_str'] != 0

tf['retweet_of'] = df['retweeted_status.id_str'].apply(conv_id)

# Remove source tweets and their retweets x days before the last day of the dataset, to fix retweet statistics, since most retweets occur x days after the original tweet
to_remove = tf.apply(retweet_fix, axis=1)  # series of (id, to_remove), which tweets need to be removed
tf.drop(to_remove[to_remove == True].index, inplace=True)
print('{} tweets removed for retweet fix'.format(len(to_remove[to_remove == True])))
del to_remove

retweet_counts = tf[tf['retweet_of'] != 0]['retweet_of'].value_counts()  # series of (id, n_retweeted), how many times did tweet with id get retweeted within the dataset
tf['retweeted_count_int'] = tf['id'].apply(get_retweet_count, args=(retweet_counts,))
del retweet_counts

tf['retweeted_count_ext'] = df['retweet_count']  # TODO currently always 0

tf['retweeted'] = tf['retweeted_count_int'] != 0


# Quote
# 'quote'==True and 'quote_of'==0 => tweet is a quote, but parent is not in dataset
# 'quote'==True and 'reweet'==True => retweet of quote

tf['quote'] = df['quoted_status.id_str'] != 0

tf['quote_of'] = df['quoted_status.id_str'].apply(conv_id)


# Reply
# 'reply'==True and 'reply_of'=0 => tweet is a reply, but parent is not in dataset

tf['reply'] = df['in_reply_to_status_id_str'] != 0

tf['reply_of'] = df['in_reply_to_status_id_str'].apply(conv_id)


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

# Convert old to new user id using transition table
def conv_user_id(id_):   
    not_found_value = 0
    return user_trans_table.get(id_, not_found_value)


# How active is the user in terms of produced statuses (tweets/retweets)
def user_activity(row):
    if row['user.account_age'] == 0:
        return None
    last_tweet_statuses_count = tf.loc[id2index(row['user.last_tweet'])]['user.statuses_count']
    return last_tweet_statuses_count / row['user.account_age']


# How active is the user in terms of statuses and likes
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

tf['user.id'] = df['user.id_str'].apply(conv_user_id)
n_not_found_ids = len(tf[tf['user.id'] == 0])
if (n_not_found_ids != 0):
    print('Warning: ' + str(n_not_found_ids) + ' user ids not found in transition table')


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

lutf = tf[['user.id', 'id']].drop_duplicates(subset='user.id', keep='last').set_index('user.id', drop=True)  # table of last tweet of every user
tf['user.last_tweet'] = tf['user.id'].apply(lambda user_id: lutf.loc[user_id]['id'])
del lutf

tf['user.statuses_count'] = df['user.statuses_count']

tf['user.favourites_count'] = df['user.favourites_count']

tf['user.account_age'] = (tf['last_day'] - pd.to_datetime(df['user.created_at'])).dt.days

tf['user.activity'] = tf.apply(user_activity, axis=1)
tf['user.mean_activity'] = tf.drop_duplicates(subset='user.id', keep='last')['user.activity'].mean()

tf['user.tweets_likes_activity'] = tf.apply(user_tweets_likes_activity, axis=1)



##############################################################################
## Save dataframe as CSV file
##############################################################################

process = psutil.Process(os.getpid())
print('Process using {} bytes of memory'.format(process.memory_info().rss))

tf.to_csv(out_path, index=False)
