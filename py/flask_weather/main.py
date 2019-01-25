import json
import urllib2
import datetime
from flask import Flask
app = Flask(__name__)


def humanRedableUNIXTime(timeStr):
    return datetime.datetime.fromtimestamp(int(timeStr)).strftime('%Y-%m-%d %H:%M:%S')

def farenheitToCelsius(farenheit):
    return (farenheit - 32)/1.8

weatherApiUrl = "https://api.forecast.io/forecast/66131959f94d7375fa3ce5d8e15da593/51.6754966,39.20888230000003"

#response = urllib2.urlopen(url)
#ans = json.loads(response.read())
#cur = ans['currently']
#print 'latitude', ans['latitude'], 'longitude', ans['longitude'], 'timezone', ans['timezone']
#print 'time', humanRedableUNIXTime(cur['time']), cur['time']
#print cur['summary'], cur['icon'], 'temp', farenheitToCelsius(float(cur['temperature']))


@app.route('/')
def index():
    response = urllib2.urlopen(weatherApiUrl)
    ans = json.loads(response.read())
    cur = ans['currently']
    return 'Weather: summary {0}, icon {1}, temp {2}'.format(cur['summary'], cur['icon'], str(farenheitToCelsius(float(cur['temperature']))))

@app.route('/hello')
def hello():
    return 'Hello'

if __name__ == '__main__':
    app.run(host="192.168.1.33", debug=True)
