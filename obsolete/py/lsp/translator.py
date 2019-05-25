
programTemplate = '''
#include <stdio.h>
#include <vector>

#include "builtins.h"

%s
'''

mainFuncTemplate = '''
int main()
{
    printf("Start....\\n");
    %s;
    return 0;
}
'''

lambdaTemplate = '''[]'''

functTemplate = '''
auto %s(%s) 
{
    %s
    return %s;
}
'''

ifTemplate = '''(%s) ? (%s) : (%s)'''

translateTable = {
        'True'    : "true",
        'False'   : "false",
        '+'       : "builtins::opAdd",
        '-'       : "builtins::opSub",
        '*'       : "builtins::opMul",
        '/'       : "builtins::opDiv",
        '>'       : "builtins::opGreaterThan",
        '<'       : "builtins::opLessThan",
        '=='      : "builtins::opEqual",
        '!='      : "builtins::opNotEqual",
        'print'   : "builtins::print",
        'println' : "builtins::println",
        'int'     : "builtins::toInt",
        'float'   : "builtins::toFloat",
        'double'  : "builtins::toDouble",
        'list'    : "builtins::createVector"
        }

lambdasCounter = 0

def argToCall(arg):
    if(isinstance(arg, list)):
        return translateToCall(arg)
    
    if(arg.isdigit()):
        return arg

    return translateTable.get(arg, arg)

def translateToCall(tree):
    if tree[0] == 'deffn':
        l = tree
        name  = l[1]
        args  = l[2]
        body  = l[3:-1]
        ret   = l[-1]
        bodyText = ';\n    '.join(map(translateToCall, body)) + ';\n    '
        retText  = argToCall(ret)
        argsText = ', '.join('auto %s' % x for x in args)
        return functTemplate % (name, argsText, bodyText, retText)

    l = map(argToCall, tree)
    if l[0] == 'define':
        return 'const auto %s = %s' % (l[1], l[2])
#    if l[0] == 'set':
        #return '%s = %s' % (l[1], l[2])
    if l[0] == 'if':
        return ifTemplate % (l[1], l[2], l[3])

    if l[0] == 'main':
        return mainFuncTemplate % ';\n    '.join(l[1:])

    return '%s(%s)' % (str(l[0]), ','.join(l[1:]))

def translate(tree):
    res = map(translateToCall, tree)
    return programTemplate % '\n    '.join(res)
