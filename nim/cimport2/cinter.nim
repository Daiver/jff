proc printf(formatstr: cstring)
    {.header: "<stdio.h>", varargs.}
printf("%s %d\n", "foo", 5)

#{.compile: "hi.c".}
proc hi*(name: cstring) {.importc.}
hi "from Nim"

proc eigen() {.importc.}
eigen()

proc addTenToSeq(input: ptr float64, size: cint, output: ptr float64) {.importc.}

proc test(arr: seq[float64]): seq[float64] = 
    var res: seq[float64] = @[]
    res.setLen(arr.len)
    addTenToSeq(cast[ptr float64](unsafeAddr(arr[0])), 
                cint(arr.len), 
                cast[ptr float64](unsafeAddr(res[0])))
    return res

echo test(@[1.0, 2, 3, 4, 5])
