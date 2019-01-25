import sequtils
import future

type 
    GainStruct* = object
        dict*: seq[float64] 

proc zeroGS *(nClasses: int):GainStruct = GainStruct(dict: newSeq[float64](nClasses))

proc fromClass *(class: int, nClasses: int): GainStruct {.inline.} = 
    assert class < nClasses
    var res = zeroGS(nClasses)
    res.dict[class] = 1
    return res

proc fromClasses *(labels: seq[int], nClasses: int): seq[GainStruct] = 
    return labels.map do (x: int) -> GainStruct:
        return fromClass(x, nClasses)

proc `+` * (a, b: GainStruct): GainStruct {.inline.} = 
    return GainStruct(dict: a.dict.zip(b.dict).map(x => x.a + x.b))

proc `-` * (a, b: GainStruct): GainStruct {.inline.} = 
    return GainStruct(dict: a.dict.zip(b.dict).map(x => x.a - x.b))

proc `+` * (a: GainStruct, b: int): GainStruct {.inline.} = 
    var res = a
    assert a.dict.len > b
    res.dict[b] += 1
    return res

proc `-` * (a: GainStruct, b: int): GainStruct {.inline.} = 
    var res = a
    assert a.dict.len > b
    res.dict[b] -= 1
    return res

proc sum *(gs: GainStruct): float64 {.inline.} = return gs.dict.foldl(a + b)

proc normalized *(gs: GainStruct): GainStruct {.inline.} = 
    let sumOfValues: float64 = gs.sum()
    var newDict = gs.dict
    for i in 0..newDict.len - 1:
        newDict[i] /= sumOfValues
    return GainStruct(dict: newDict)

proc giniImp *(gs: GainStruct): float64 = 
    let gsNorm = gs.normalized
    var sum = gsNorm.dict.map(x => x * x).foldl(a + b)
    return 1.0 - sum

proc gainFunc *(gAll, gL, gR: GainStruct): float64 {.procvar.} = 
    let gAllSum = gAll.sum
    return gAll.giniImp - gL.sum/gAllSum * gL.giniImp - gR.sum/gAllSum * gR.giniImp
