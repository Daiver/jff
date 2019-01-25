cinter:
	gcc -c hi.c
	ar rvs mylib.a hi.o 
	nim c --passL:mylib.a --passL:-llapacke cinter.nim
