# -*- coding: utf-8 -*-

import json

from geocoders.yandex import geocoder
from TwitterAPI import TwitterAPI

def extractDataFromCity(city, api, geocode):
    geo_data = geocode(city)
    print geo_data[0], geo_data[1]
    users_in_city = set()
    #all_tweets = []
    for i in xrange(1):
        r = api.request('search/tweets', 
                {'geocode':'%f,%f,10mi' % geo_data[1], 'count' : 100})
        print r.status_code
        tweets = list(r.get_iterator())
        for item in tweets: 
            users_in_city |= {item['user']['screen_name']}
        #all_tweets += tweets
        print len(tweets)
        print('QUOTA: %s' % r.get_rest_quota())
    print len(users_in_city)
    return geo_data, list(users_in_city)

def getUsersByCity(api, geocode):
    cities = [
                  'Москва'
                , 'Воронеж'
                , 'Нижний Новгород'
                , 'Казань'
                , 'Санкт-Петербург'
                , 'Ростов'
            ]
    with open('city_tweet_data', 'w') as f:
        data = map(lambda c: extractDataFromCity(c, api, geocode), cities)
        f.write(json.dumps(data))

if __name__ == '__main__':
    c_key, c_secret, a_key, a_secret, ya_key = open('key_data').read().split('\n')[:5]
    #geocode = geocoder(ya_key)
    api = TwitterAPI(c_key, c_secret, a_key, a_secret)
    print api

    with open('city_tweet_data') as f:
        data = json.loads(f.read())

    flag = False
    with open('tweets_data', 'a+') as f: 
        for i, city in enumerate(data):
            print city[0][0], city[0][1]
            for user in city[1]:
                print user
                if user == 'semi_con_ductor': flag = True
                if not flag: continue
                r = api.request('statuses/user_timeline',
                        {'screen_name': user, 'count' : 200})
                print r.status_code
                tweets = list(r.get_iterator())
                for item in tweets:
                    f.write('%d %s %s %s\n' % 
                            (i, json.dumps(city[0][0]), user, json.dumps(item['text'])))
                print len(tweets)
                print('QUOTA: %s' % r.get_rest_quota())
