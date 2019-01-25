import strutils

var memory : array[300, int]
for i in countup(0, 299):
    memory[i] = 0

var memPointer = 0

#let line = stdin.readLine
let fName = "tmp.bf"
let text = readFile(fName)

var brc = 0
var j   = 0
for x in text:
    #x = text[i]
    case x 
    of '+':
        memory[memPointer] += 1
    of '-':
        memory[memPointer] -= 1
    of '>':
        memPointer += 1
    of '<':
        memPointer -= 1
    of '.':
        stdout.write chr(memory[memPointer])
    of '[':
        if memory[j] != 0:
            continue
        brc += 1
        #while brc > 0:
            

    of ']':
        discard
    else:
        discard
    #echo mem

