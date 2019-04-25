globalDict = {
        '+' : lambda l: l[0] + l[1],
        '-' : lambda l: l[0] - l[1],
        '*' : lambda l: l[0] * l[1],
        '/' : lambda l: l[0] / l[1]
        }

def execArg(arg):
    if isinstance(arg, list):
        return execute(arg)
    if arg.isdigit():
        return int(arg)
    return globalDict[arg]

def execute(tree):
    #f = funcs[tree[0]]
    l = map(execArg, tree)
    return l[0](l[1:])
