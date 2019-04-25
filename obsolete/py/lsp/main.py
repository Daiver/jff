import lexer
import parser
import executer
import translator
import os

s = '''
(deffn f 
    (x y) 
    (define a (+ x y))
    (define b (- x y))
    (define c (* a b))
    c)

(deffn abs
    (x)
    (if (> x 0) x (- 0 x)))

(main
    (println (abs 1))
    (println (abs (- 0 2)))
    (println True)
    (println (> 1 2))
    (println (== 0 1))
    (println (== 0 0))
    (define d 1)
)
'''

def filterTree(tree):
    if not isinstance(tree, list):
        return tree
    tmp = filter(lambda x: x != '' and x != '\n', tree)
    return map(filterTree, tmp)

if __name__ == '__main__':

    t = lexer.getTokens(s)
    tree = parser.parse(t)[0]
    tree = filterTree(tree)
    print tree
    src = translator.translate(tree)
    print src
    with open('tmp.cpp', 'w') as f:
        f.write(src)
    os.system("g++ -std=c++14 tmp.cpp && ./a.out")
    #print executer.execute(tree)
