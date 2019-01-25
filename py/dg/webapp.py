from flask import Flask
from flask import request
app = Flask(__name__)
import dg
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

@app.route("/search/", methods=['POST', 'GET'])
def resultPage():
    try:
        ans = dg.activate(dht, request.form['q'])
        if len(ans) == 0:
            return head + '<h2 align=center>No results</h2>'
        return css+ (head % request.form['q']) + ''.join(
                map(lambda x: """
                    <div style='position:absolute ;left:300px'>
                        <a href='%s'>%s : %d</a>
                    </div></br>\n
                    """ % (x[1], x[1], x[0]), ans)
        ) + '</body></html>'
    except Exception as e:
        return str(e)

@app.route("/")
def indexPage():
    return '</br></br></br>' + (head % '') + ''
if __name__ == "__main__":
    #app.run()
    app.run("127.0.0.1", 8080)
