proc printf(formatstr: cstring)
    {.header: "<stdio.h>", varargs.}
printf("%s %d\n", "foo", 5)

{.compile: "hi.c".}
proc hi*(name: cstring) {.importc.}
hi "from Nim"
