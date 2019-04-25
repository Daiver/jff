from flask import Flask
from flask import request
app = Flask(__name__)
import dht as pydht

dht = pydht.DHT("127.0.0.1", 6000, "127.0.0.1", 8088)
head = '''
        <h2 align=center>Search</h2>
        <form action="/search/" method="post" align=center>
            <input name="q" value=%s></input>
            <input type=submit />
        </form></br>
    '''

css = '''
    <html>
        <head>
        </head>
    <body>
'''

@app.route('/dn/<page>')
def show_user_profile(page):
    # show the user profile for that user
    if page not in dht:
        return "<h2>404</h2><h3>%s not found</h3>" % page
    return dht[page]

if __name__ == "__main__":
    #app.run()
    app.run("127.0.0.1", 8080)
