#from hyperopt import fmin, tpe, hp, rand
import hyperopt
def fn(params):
    print params
    return params['x'] ** 2 + int(params['c'])
best = hyperopt.fmin(fn=fn,
            space={
                'x' : hyperopt.hp.uniform('x', -1, 1),
                'c' : hyperopt.hp.choice('c', [5, 4]),
            },
            algo=hyperopt.tpe.suggest,
            #algo=rand.suggest,
            max_evals=20,
            verbose=True,
            return_argmin=True)
print 'BEST'
print best
best['c'] = [5, 4][best['c']]
print 'ERROR', fn(best)
