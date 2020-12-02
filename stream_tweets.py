# -*- coding: utf-8 -*-
from configparser import ConfigParser
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from urllib3.exceptions import ProtocolError
from tweepy.streaming import StreamListener
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError
import json
import time
import logging
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
import pandas as pd
import os
import datetime
os.chdir(os.path.dirname(__file__))

#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")

pd.set_option("display.max_columns", 10)

session_info = config_object["SESSION_INFO"]
twitter_config = config_object["TWITTER_CONFIG"]
bigquery_config = config_object["BIGQUERY_CONFIG"]

max_tweets = session_info["max_tweets"]
db_name = session_info['session_name']
account_name = bigquery_config["account_name"]
bq_auth = bigquery_config["google_credential_path"]

session_id = account_name + '_' + db_name

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_auth
if not os.path.exists(os.path.join(os.path.dirname(__file__),db_name)):
    os.makedirs(db_name)

logging.basicConfig(filename=os.path.join(db_name,'tweets_capture.log'),level=logging.INFO)

client = google.cloud.logging.Client()
handler = CloudLoggingHandler(client, name=session_id)
logging.getLogger().setLevel(logging.INFO) # defaults to WARN
setup_logging(handler)

log_name = '_' + db_name + '_to_process.txt'
error_file_log_name = '_' + db_name + '_error_files.txt'

log_path = os.path.join(db_name,log_name)
error_file_log_path = os.path.join(db_name,error_file_log_name)

db_backlog = open(os.path.join(db_name,log_name),'a+')
db_backlog.close()
error_file_log = open(os.path.join(db_name,error_file_log_name),'a+')
error_file_log.close()

# inherit from StreamListener class
class SListener(StreamListener):
    def __init__(self, api = None, fprefix = 'streamer'):
        # define the filename with time as prefix
        self.api = api or API()
        self.counter = 0
        self.max_counter = 0
        self.fprefix = fprefix
        self.output  = open(os.path.join('%s', '%s_%s.json') % (db_name, self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
        # self.db_backlog_name = self.output.name.replace('.json', '.txt')
        self.out_count = 1
    def on_data(self, data):
        if  'in_reply_to_status' in data:
            self.on_status(data)
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return
    def on_status(self, status):
        # if the number of tweets reaches 20000
        # create a new file
        self.output.write(status)
        self.counter += 1
        self.max_counter += 1
        print(f'{self.max_counter} total tweets')
        if self.counter >= 100:
            # tweet_autodb.main(self.output.name, 'test_db')
            with open(log_path, 'a') as bl:
                bl.write(self.output.name + '\n')
            self.output.close()
            self.output = open(os.path.join('%s', '%s_%s.json') % (db_name, self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0
            self.out_count += 1
        return
    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            cdate = "Error code 420 at:"+str(datetime.datetime.now())
            print(cdate)
            logging.info(cdate)
            logging.info("Sleeping for 15 mins")
            time.sleep(900)
            return False

def main():
    consumer_key = twitter_config["ckey"]
    consumer_secret = twitter_config["csecret"]
    auth = OAuthHandler(consumer_key, consumer_secret)
    # access key authentication
    access_token = twitter_config['token']
    access_token_secret = twitter_config['token_secret']
    auth.set_access_token(access_token, access_token_secret)
    # set up the API with the authentication handler
    api = API(auth)
    # set up words to hear
    keywords_to_hear = session_info["keywords"]
    keywords_to_hear = keywords_to_hear.split(',')
    # instantiate the SListener object
    listen = SListener(api)
    # instantiate the stream object
    stream = Stream(auth, listen)

    #send_db = Popen(['python3', '/Users/p-73/Documents/Data Science/Tweet Sentiment/build1/tweet_autodb.py', log_name, db_name, error_file_log_name])
    # begin collecting data
    while listen.max_counter <= int(max_tweets):
        # maintain connection unless interrupted
        try:
            msg = f"[STREAM] Stream starting...\nSession: {session_id}\nKeywords: {keywords_to_hear}"
            logging.info(msg)
            stream.filter(languages=["ja"], track=keywords_to_hear)
        # reconnect automantically if error arise
        # due to unstable network connection
        except (ProtocolError, AttributeError, Timeout, ReadTimeoutError, ConnectionError):
            msg = "[STREAM] Stream stopped! Reconnecting to twitter stream"
            logging.info(msg)
            continue

if __name__ == '__main__':
    main()
