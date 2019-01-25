import elasticsearch
from elasticsearch import Elasticsearch

from datetime import datetime

if __name__ == '__main__':
    es = Elasticsearch([{'host' : "127.0.0.1", 'port' : 9200}])

    doc = {
	'author': 'kimchy4',
	'text': 'bonsai cool.',
        'crop' : False,
	'timestamp': datetime.now(),
    }
    res = es.index(index="test-index", doc_type='tweet', body=doc)
    print(res['created'])

#    try:
        #res = es.get(index="test-index", doc_type='tweet', id='AVPI3wULldBPvsHqg1k2')
        #print(res['_source'])
    #except elasticsearch.exceptions.NotFoundError as e:
        #print 'Cannot find', e

    #print 'exists?', es.exists(index="test-index", doc_type='tweet', id=2)
    #print 'exists?', es.exists(index="test-index", doc_type='tweet', id='AVPI3wULldBPvsHqg1k2')
    es.indices.refresh(index="test-index")
    res = es.search(
            index="test-index", doc_type='tweet', body={"query": {
                    "match": {"crop" : False}
                }}
            )
    print 'Search res', res['hits']['total']

    res = es.search(
            index="test-index", doc_type='tweet', body={"query": {"match_all": {}}}
            )
    print res
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
	print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
    #print res['hits']['total']
