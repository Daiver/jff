import sacred
from sacred.observers import MongoObserver

ex = sacred.Experiment('ex')
ex.observers.append(MongoObserver.create(db_name='db'))

@ex.config
def cfg():
    x = 120

@ex.main
def main1(x):
    print x

ex.run()
ex.run(config_updates={'x' : 1})
