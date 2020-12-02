from configparser import ConfigParser
import sqlalchemy
import logging
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
import time
import datetime
import os
from pathlib import Path
import traceback
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas_gbq
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.engine.url import make_url
from sqlalchemy_utils.functions import database_exists
from sqlalchemy_utils.functions import create_database
import glob
from advertools import extract_emoji
import re
import string
import sys

## TO DO ##
# crete new script, create new table with tweet id and prediction, add flag to tweet table for check,
# batch query rows in tweet table with no flag, apply predictions and write to new table
##########


# # create a new database
# engine = sqlalchemy.create_engine('postgresql://postgres:principle19@localhost:5432')
# # connect to the database
# con = engine.connect()
# # end open transaction
# con.execute('commit')
# # # create new databse my_databse
# con.execute('create database tweet_ryzen')
# con.close()

# specify table
# engine = sqlalchemy.create_engine('postgresql://postgres:principle19@localhost:5432/tweet_ryzen')

# -*- coding: utf-8 -*-


def list_col_to_str_col(df):
    """
    if a column in the dataframe is a list
    convert it into str so it can be stored in the database
    Parameters
    ----------
        df: the dataframe to be converted
    Returns
    -------
        None
    """
    if len(df) > 0:
        for col in df.columns:
            if type(df[col].iloc[0]) is list:
                df[col] = df[col].apply(str)

def expand_dict(df):
    """
    expand a two column dataframe of which the second column
    contains dictionarys
    Parameters
    ----------
        df: 2-column pandas dataframe
    Returns
    -------
        df_with_id: multi-column pandas dataframe
            the number of columns after expantion is equal to the number
            of keys in the dictionary
    """
    ID = df.iloc[:,0]
    series = df.iloc[:,1].tolist()
    df_from_series = pd.DataFrame(series, index=df.index)
    df_with_id = pd.concat([ID, df_from_series], axis=1)

    return df_with_id

def drop_empty_list(df):
    """
    replace the empty lists in a two-column dataframe into np.nan
    so they can be dropped using df.dropna() method
    Parameters
    ----------
        df: two-column pandas dataframe
    Returns
    -------
        df_dropped: with all the empty lists dropped
    """

    replace_nan = lambda x: np.nan if len(x)==0 else x
    col = df.iloc[:,1].copy()
    new_col = col.apply(replace_nan)
    df_replaced = pd.concat([df.iloc[:,0], new_col], axis=1)

    df_dropped = df_replaced.dropna()
    return df_dropped


def merge_dicts(df):
    """
    when the other column of the dataframe contains a list of dictionarys
    with the same key and value pairs
    we merge the list to a single dictionary
    Parameters
    ----------
        df: input dataframe
    Returns
    -------
        df_merged_w_id
    """
    series = df.iloc[:,1].copy().tolist()
    merged_series = []
    for dict_lst in series:
        dict_1 = dict_lst[0]
        total_dict = {key:[] for key in list(dict_1.keys())}
        for dit in dict_lst:
            for key in list(dict_1.keys()):
                total_dict[key].append(dit[key])
        merged_series.append(total_dict)
    df_merged = pd.DataFrame(merged_series, index=df.index)
    df_merged_w_id = pd.concat([df.iloc[:,0], df_merged], axis=1)

    return df_merged_w_id

#%%
def json2df(filename):
    """
    load json file into a list of dictionarys

    Parameters
    ----------
        filename: string
            the directory of the json file

    Returns
    -------
        to_sql: list of pandas dataframes
            each dataframe corresponds to a table in the database
    """


    """
    First section:
        load the tweets from the json file
        then coarsely split them into base tweets and dict tweets
    """

    # convert from JSON to Python object


    # initialize a list for all the json lines
    tweets = []

    # initialize a dictionary for all the dataframes
    to_sql = {}

    # load the json file into a list
    with open(filename, 'r') as f:
        for line in f:
            if len(line) > 20:
                tweet = json.loads(line)
                tweets.append(tweet)

    # convert the list of lines into a dataframe
    tweets_df = pd.DataFrame(tweets)
    #print(tweets_df.head())
    tweets_df.rename(columns={'id':'tweet_id',
                              'id_str':'tweet_id_str'}, inplace=True)

    # print(tweets_df.head())
    # specify the columns that have to be stored in dicts
    to_dicts = ['tweet_id',
                'coordinates',
                'entities',
                'extended_entities',
                'extended_tweet',
                'place',
                'quoted_status',
                'quoted_status_permalink',
                'retweeted_status',
                'user']

    # specify the columns to drop
    to_drop = [ 'contributors',
                'display_text_range',
                'coordinates',
                'entities',
                'extended_entities',
                'extended_tweet',
                'geo',
                'place',
                'quoted_status',
                'quoted_status_permalink',
                'retweeted_status',
                'user']


    # divide the raw tweets into normal part (without dicts)
    # and dict part (needs to be saved into multiple tables)
    #tweets_dicts = tweets_df[to_dicts]
    tweets_dicts = tweets_df.loc[:,tweets_df.columns.isin(to_dicts)]
    tweets_non_dicts = tweets_df.drop(to_drop, axis=1, errors='ignore')

    # add the base tweets into the dictionary
    to_sql['base_tweets'] = tweets_non_dicts

    # seperate the columns in the dicts dataframe into multiple dataframes
    tweets_2dict_lst = []
    tweets_2dict_lst_key = {}
    for idx in range(1, len(to_dicts)):
        if to_dicts[idx] in [col for col in tweets_df.columns]:
            tweets_2dict_lst_key[to_dicts[idx]] = len(tweets_2dict_lst)
            tweets_2dict_lst.append(tweets_dicts[['tweet_id', to_dicts[idx]]].dropna())

    """
    Second section:
        for each dataframes in the tweets_2dict_lst
        flatten them into a cleaner format of dataframes
    """

    """
    2.1 coordinates
    """
    # if 'coordinates' in [col for col in tweets_df.columns]:
    #     col_i = df.columns.get_loc("coordinates")
    coord_idx = tweets_2dict_lst_key['coordinates']
    coor_df = tweets_2dict_lst[coord_idx]

    if len(coor_df)==0:

        pass

    else:

        coor_df = expand_dict(coor_df)

        list_col_to_str_col(coor_df)

        to_sql['coordinates'] = coor_df

    """
    2.2 entities
    """
    # get the entities first
    entities_idx = tweets_2dict_lst_key['entities']
    entities_df = expand_dict(tweets_2dict_lst[entities_idx])

    # get hashtags and user_mentions
    hashtags_df = drop_empty_list(entities_df[['tweet_id', 'hashtags']])
    user_mentions_df = drop_empty_list(entities_df[['tweet_id', 'user_mentions']])

    # merge them
    hashtags_df = merge_dicts(hashtags_df)
    user_mentions_df = merge_dicts(user_mentions_df)

    # convert list object to str
    list_col_to_str_col(hashtags_df)
    list_col_to_str_col(user_mentions_df)

    # add into list
    to_sql['hashtags'] = hashtags_df.rename({'text':'hashtag'}, axis=1)
    to_sql['user_mentions'] = user_mentions_df

    """
    2.3-2.5 extended tweet, quoted tweet, retweeted
    """

    # use the json indexing to get the useful information
    # initialize each dictionary
    extended_tweet = {'tweet_id':[],
                      'full_text':[],
                      'user_mentions':[],
                      'extended_hashtags':[]}

    quoted_tweet = {'tweet_id':[],
                    'quoted_id':[],
                    'quoted_text':[],
                    'quoted_hashtags':[]}

    quoted_user = {'tweet_id':[],
                   'quoted_user':[]}

    retweeted_tweet = {'tweet_id':[],
                       'retweeted_id':[],
                       'retweeted_text':[],
                       'retweeted_hashtags':[]}

    retweeted_user = {'tweet_id':[],
                      'retweeted_user':[]}


    with open(filename, 'r') as file:
        for line in file:
            if line != '\n':
                tweet = json.loads(line)

                # get info for extended tweets
                if 'extended_tweet' in tweet.keys():
                    extended_tweet['tweet_id'].append(tweet['id'])
                    extended_tweet['full_text'].append(tweet['extended_tweet']['full_text'])
                    user_mentions = [dit['id'] for dit in tweet['extended_tweet']['entities']['user_mentions']]
                    extended_tweet['user_mentions'].append(user_mentions)
                    extended_hashtags = [dit['text'] for dit in tweet['extended_tweet']['entities']['hashtags']]
                    extended_tweet['extended_hashtags'].append(extended_hashtags)

                # get info for quoted tweets
                if 'quoted_status' in tweet.keys():
                    quoted_tweet['tweet_id'].append(tweet['id'])
                    quoted_tweet['quoted_id'].append(tweet['quoted_status']['id'])
                    quoted_tweet['quoted_text'].append(tweet['quoted_status']['text'])
                    quoted_hashtags = [dit['text'] for dit in tweet['quoted_status']['entities']['hashtags']]
                    quoted_tweet['quoted_hashtags'].append(quoted_hashtags)

                    # get info for quoted users

                    quoted_user['tweet_id'].append(tweet['id'])
                    quoted_user['quoted_user'].append(tweet['quoted_status']['user'])

                # get info for retweeted tweets
                if 'retweeted_status' in tweet.keys():
                    retweeted_tweet['tweet_id'].append(tweet['id'])
                    retweeted_tweet['retweeted_id'].append(tweet['retweeted_status']['id'])
                    retweeted_tweet['retweeted_text'].append(tweet['retweeted_status']['text'])
                    retweeted_hashtags = [dit['text'] for dit in tweet['retweeted_status']['entities']['hashtags']]
                    retweeted_tweet['retweeted_hashtags'].append(retweeted_hashtags)

                    # get info for quoted users

                    retweeted_user['tweet_id'].append(tweet['id'])
                    retweeted_user['retweeted_user'].append(tweet['retweeted_status']['user'])

        extended_tweet_df = pd.DataFrame(extended_tweet)
        quoted_tweet_df = pd.DataFrame(quoted_tweet)
        quoted_user_df = pd.DataFrame(quoted_user)
        retweeted_tweet_df = pd.DataFrame(retweeted_tweet)
        retweeted_user_df = pd.DataFrame(retweeted_user)

    # handle the new dataframes
    # expand two user dataframes
    quoted_user_df = expand_dict(quoted_user_df)
    retweeted_user_df = expand_dict(retweeted_user_df)

    # convert list to str
    list_col_to_str_col(extended_tweet_df)
    list_col_to_str_col(quoted_tweet_df)
    list_col_to_str_col(retweeted_tweet_df)

    # add into list
    to_sql['extended_tweets'] = extended_tweet_df
    to_sql['quoted_tweets'] = quoted_tweet_df
    to_sql['quoted_user'] = quoted_user_df
    to_sql['retweeted_tweet'] = retweeted_tweet_df
    to_sql['retweeted_user'] = retweeted_user_df

    """
    2.6 place
    """
    # handle place dataframe
    place_idx = tweets_2dict_lst_key['place']
    place_df = tweets_2dict_lst[place_idx]

    if len(place_df)==0:

        #to_sql['place'] = pd.DataFrame()
        pass
    else:

        # expand it first
        place_df = expand_dict(place_df).drop(['attributes'], axis=1)

        # note that bounding box column contains dictionarys
        # make it a new dataframe into two columns
        bounding_box_lst = place_df.loc[:, 'bounding_box'].copy().tolist()
        bounding_box_df = pd.DataFrame(bounding_box_lst,
                                       index=place_df.index)
        bounding_box_df.rename(columns={'coordinates': 'bounding_box_coordinates',
                                        'type': 'bounding_box_type'})

        # then merge it back and drop the original one
        place_df = pd.concat([place_df, bounding_box_df], axis=1)
        place_df.drop(['bounding_box'], axis=1, inplace=True)

        # convert list to str
        list_col_to_str_col(place_df)

        # add to list
        to_sql['place'] = place_df

    """
    2.7 user
    """
    # expand it
    user_idx = tweets_2dict_lst_key['user']
    user_df = expand_dict(tweets_2dict_lst[user_idx])
    to_keep = ['tweet_id',
               'created_at',
               'description',
               'favourites_count',
               'followers_count',
               'geo_enabled',
               'id',
               'lang',
               'location',
               'url',
               'verified',
               'friends_count']

    # add to list
    to_sql['tweet_user'] = user_df[to_keep].copy()

    """
    Third section
        parse datetime object
    """
    # for each dataframe in the list
    for df_name, df in to_sql.items():
        if len(df) != 0:
            if 'created_at' in df.columns:
                df.created_at = pd.to_datetime(df.created_at)
            df.drop_duplicates(inplace=True)
            if df_name == 'base_tweets':
                try:
                    print('processing tweet...')
                    df['clean_text'] = df['text'].apply(processTweet)
                    # print('predicting sentiment...') # moved to separate process
                    # df['sentiment_prediction'] = df['clean_text'].apply(predict_bert)
                except Exception as e:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)
                    print(traceback.format_exc())
                    logging.info(message)
            elif df_name == 'extended_tweets':
                try:
                    print('processing tweet...')
                    df['clean_text'] = df['full_text'].apply(processTweet)
                    # print('predicting sentiment...') # moved to separate process 
                    # df['sentiment_prediction'] = df['clean_text'].apply(predict_bert)
                except Exception as e:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)
                    print(traceback.format_exc())
                    logging.info(message)
    return to_sql

#%%
# create a new database
#engine = create_engine('postgresql://postgres:461022@localhost:5432')
#
#con = engine.connect()
#con.execute('commit')
#con.execute('create database tweets_48')
#con.close()

#%%
# to postgresql
#engine = create_engine('postgresql://postgres:461022@localhost:5432/tweets_48')
# call the function to load the json into dataframes



def processTweet(tweet):
    emojis = extract_emoji(tweet)
    emojis = emojis['emoji_flat']
    emojis = ''.join(emojis)
    tweet = tweet.lower() # convert text to lower-case
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
    tweet = re.sub('\n+', '', tweet) # remove new line
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = re.sub('[^\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A]+', '', tweet) # remove japanese punctuation
    tweet = tweet + emojis
    return tweet

def main():
    os.chdir(os.path.dirname(__file__))
    #Read config.ini file
    config_object = ConfigParser()
    config_object.read("config.ini")

    session_info = config_object["SESSION_INFO"]
    postgres_config = config_object["POSTGRES_CONFIG"]
    bigquery_config = config_object["BIGQUERY_CONFIG"]
    
    db_type = session_info["database"]
    db_name = session_info["session_name"]

    psql_server = postgres_config["host"]
    psql_port = postgres_config["port"]
    psql_user = postgres_config["username"]
    psql_password = postgres_config["password"]

    bq_project = bigquery_config["project_name"]
    bq_account = bigquery_config["account_name"]
    bq_auth = bigquery_config["google_credential_path"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_auth

    session_id = bq_account + '_' + db_name

    log_name = '_' + db_name + '_to_process.txt'
    error_file_log_name = '_' + db_name + '_error_files.txt'

    backlog_file = os.path.join(db_name,log_name)
    error_file_log = os.path.join(db_name,error_file_log_name)

    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client, name=session_id)
    logging.getLogger().setLevel(logging.INFO)  # defaults to WARN
    setup_logging(handler)

    print(f'Begin:\nProcessing json files in {log_name}\n{db_type}:\nProject: {bq_project}\nDataset: {bq_account}.{db_name}')
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(db_name,'tweets_processing.log'),level=logging.INFO)


    while True:
        while Path(backlog_file).stat().st_size == 0:
            wait_msg = f'WAITING {str(datetime.datetime.now())}: waiting for json data'
            print(wait_msg)
            logging.info(wait_msg)
            time.sleep(60)
        if Path(backlog_file).stat().st_size > 0:
            with open(backlog_file, 'r') as f:
                lines = f.readlines()
                file_raw = lines[0]
                file = file_raw.strip('\n')
                with open(backlog_file, 'w') as update:
                    for line in lines:
                        if line != file_raw:
                            update.write(line)
                if os.path.isfile(file):
                    check_conn = False
                    
                    while check_conn == False:
                        try:
                            msg = f'Connecting to {db_type}...'
                            print(msg)
                            logging.info(msg)
                            if db_type =='postgres':
                                url = make_url(f'postgresql://{psql_user}:{psql_password}@{psql_server}:{psql_port}/{db_name}')
                                engine = sqlalchemy.create_engine(url)
                                with engine.connect() as con:
                                    rs = con.execute('SELECT 1')
                                    print(rs)
                                check_conn = True
                                msg = 'Postgres connection successful!'
                                logging.info(msg)
                                print(msg)
                                if not database_exists(engine.url):#db does not exist
                                    # create a new database
                                    create_database(engine.url)
                                    msg = f"{db_name} postgres database not found. Creating new database."
                                    logging.info(msg)
                                    print(msg)
                                    
                            elif db_type == 'bigquery':
                                project_name = bq_project
                                dataset_name = bq_account + '_' + db_name
                                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_auth
                                client = bigquery.Client()
                                # set pandas_gbq project
                                pandas_gbq.context.project = project_name

                                # Set dataset_id to the ID of the dataset to determine existence.
                                dataset_id = project_name + '.' + dataset_name

                                try:
                                    client.get_dataset(dataset_id)  # Make an API request.
                                    msg = "BigQuery: Dataset {} already exists".format(dataset_id)
                                    print(msg)
                                    logging.info(msg)
                                    check_conn = True
                                except NotFound:
                                    msg = "BigQuery: Dataset {} is not found. Creating dataset.".format(dataset_id)
                                    print(msg)
                                    logging.info(msg)
                                    dataset = bigquery.Dataset(dataset_id)
                                    # TODO(developer): Specify the geographic location where the dataset should reside.
                                    dataset.location = "US"

                                    # Send the dataset to the API for creation, with an explicit timeout.
                                    # Raises google.api_core.exceptions.Conflict if the Dataset already
                                    # exists within the project.
                                    dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
                                    msg = "BigQuery: Created dataset {}.{}".format(client.project, dataset.dataset_id)
                                    logging.info(msg)
                                    print(msg)

                        except Exception as e:
                            # error_msg = f'ERROR {str(datetime.datetime.now())}: Connection to postgres could not be established'
                            # logging.info(error_msg)
                            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                            message = template.format(type(e).__name__, e.args)
                            print(message)
                            print(traceback.format_exc())
                            logging.info(message)
                            check_conn = False
                            with open(error_file_log,'a+') as f:
                                f.write(file_raw)
                            sys.exit()
                    try:
                        file_lst = glob.glob(file)
                        for file_idx, file_name in enumerate(file_lst):
                            if file_idx == 0:
                                option = 'append'
                            else:
                                option = 'append'
                            process_start_msg = f'PROCESSING: {str(datetime.datetime.now())}: processing {file}'
                            print(process_start_msg)
                            logging.info(process_start_msg)
                            try:
                                to_sql_g = json2df(file_name)
                                print('Conversion of json to df complete.')
                            except Exception as e:
                                template = "An exception of type {0} occurred during json to df conversion or prediction. Arguments:\n{1!r}"
                                message = template.format(type(e).__name__, e.args)
                                print(message)
                                print(traceback.format_exc())
                                logging.info(message)
                            else:
                                for df_name, df in to_sql_g.items():
                                    table_id = project_name + '.' + dataset_name + '.' + df_name
                                    msg = f'{str(datetime.datetime.now())}: uploading {df_name}  -->  {db_type}:{table_id}'
                                    print(msg)
                                    logging.info(msg)
                                    if db_type == 'postgres':
                                        df.to_sql(df_name, con=engine, if_exists=option)
                                    elif db_type == 'bigquery':
                                        table_name = df_name
                                        # create new table and write df
                                        dest_table = dataset_name + '.' + table_name
                                        pandas_gbq.to_gbq(df, dest_table, if_exists='append')
                                        msg = f'{str(datetime.datetime.now())}: uploading {df_name}  -->  {db_type} complete'
                                        print(msg)
                                        logging.info(msg)

                                if db_type == 'postgres':
                                    engine.dispose()
                                cdate = str(datetime.datetime.now())
                                success_msg = 'SUCCESS {}: (DB {}) FILE {}'.format(cdate, db_type, file_name)
                                logging.info(success_msg)
                    except Exception as e:
                        #cdate = str(datetime.datetime.now())
                        #error_msg = 'ERROR {}: FILE {}: {}'.format(cdate, file_name, e)
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(e).__name__, e.args)
                        print(message)
                        print(traceback.format_exc())
                        logging.info(message)
                        with open(error_file_log,'a+') as f:
                            f.write(file_raw)
                else:
                    msg = f'{file} does not exist'
                    print(msg)
                    logging.info(msg)
                    with open(error_file_log,'a+') as f:
                        f.write(file_raw)
                    sys.exit()

if __name__ == '__main__':
    main()
