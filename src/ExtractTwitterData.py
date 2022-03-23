import sys
import requests
import os
import pandas as pd
import csv
import time
from datetime import datetime
from datetime import timedelta
import emoji
import re

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: ExtractTwitterData <Output_data_file_location>', file=sys.stderr)
        exit(-1)

# Create file
csvFile = open(sys.argv[1] + '/data.csv', 'a', newline='', encoding='utf-8')
csvWriter = csv.writer(csvFile)

# Create headers for the data you want to save, in this example, we only want save these columns in our dataset
csvWriter.writerow(['id', 'like_count', 'retweet_count', 'tweet'])
csvFile.close()

# Twitter API Token
os.environ['TOKEN'] = ''  # place your own generated twitter api token
date_format_str = '%Y-%m-%dT%H:%M:%S.%fZ'
word_list = ['covid-19', 'covid19', 'covid_19', 'covid']


def append_to_csv(json_response):
    # A counter variable
    counter = 0

    # Open OR create the target CSV file
    csvFile = open(sys.argv[1] + '/data.csv', 'a', newline='', encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    # Loop through each tweet
    for tweet in json_response['data']:
        # Tweet ID
        tweet_id = tweet['id']

        # Tweet metrics, like count
        like_count = tweet['public_metrics']['like_count']

        # Tweet metrics, retweet_count
        retweet_count = tweet['public_metrics']['retweet_count']

        # Tweet text
        text = tweet['text']

        # Assemble all data in a list
        res = [tweet_id, like_count, retweet_count, text]

        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)


# Return twitter api token
def auth():
    return os.getenv('TOKEN')


# create header for twitter request
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


# create url with query params
def create_url(keyword, start_date, end_date, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'tweet.fields': 'id,text,public_metrics',
                    'place.fields': 'country_code',
                    'next_token': {}}
    return search_url, query_params


# request to twitter api
def connect_to_endpoint(url, headers, params, next_token=None):
    params['next_token'] = next_token  # params object received from create_url function
    response = requests.request("GET", url, headers=headers, params=params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# Inputs for the request
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "covid-19 -is:retweet lang:en"
max_results = 100
start_time = '2021-10-12T00:00:00.000Z'
end_time = '2021-10-12T00:30:00.000Z'

# Get tweets
for i in range(10):
    url = create_url(keyword, start_time, end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    append_to_csv(json_response)
    # update start time = end time + 1sec
    start_time = (datetime.strptime(end_time, date_format_str) + timedelta(seconds=1)).strftime(date_format_str)
    # update end time = end time + 30 min
    end_time = (datetime.strptime(end_time, date_format_str) + timedelta(minutes=30)).strftime(date_format_str)
    # sleep for some seconds, as twitter api has a limit on number of requests in 15 minutes
    time.sleep(2)


def remove_emoji(text):
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI.get('en')]
    clean_text = ' '.join([st for st in text.split() if not any(i in st for i in emoji_list)])
    return clean_text


def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    return text


def remove_usernames(text):
    text = re.sub(r'()@\w+', r'\1', text)
    return text


def remove_word(text):
    for word in word_list:
        text = re.compile(r'\b(' + word + r')\b', flags=re.IGNORECASE).sub('', text)
    return text


def keep_alpha_whitespace(text):
    text = re.sub(r'[^A-Za-z ]+', '', text)
    return text


def remove_multiple_whitespace(text):
    text = re.compile(r'\s+').sub(' ', text)
    return text


# Read the data file
df = pd.read_csv(sys.argv[1] + '/data.csv')

# Clean tweets
df['tweet'] = df['tweet'].apply(remove_emoji)                   # remove emoji
df['tweet'] = df['tweet'].apply(remove_url)                     # remove urls
df['tweet'] = df['tweet'].apply(remove_usernames)               # remove usernames (words starting with @)
df['tweet'] = df['tweet'].apply(remove_word)                    # remove specific words
df['tweet'] = df['tweet'].apply(keep_alpha_whitespace)          # remove all characters except characters and whitespace
df['tweet'] = df['tweet'].apply(remove_multiple_whitespace)     # remove multiple whitespaces
df['tweet'] = df['tweet'].str.lower()                           # convert all data to lower case

# drop rows containing an values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# save final data
df.to_csv(sys.argv[1] + '/final_twitter_data.csv', header=False, index=False, columns=['id', 'tweet'])
