clean: .PHONY
	rm build/clibrary.a build/lapacke_backend.o

build/clibrary.a:
	gcc -c lapacke_backend.c -o build/lapacke_backend.o
	ar rvs build/clibrary.a build/lapacke_backend.o

tests: build/clibrary.a
	nim c --passL:build/clibrary.a --passL:-llapacke tests.nim

linreg: build/clibrary.a
	nim c --passL:clibrary.a --passL:-llapacke linreg.nim

main: build/clibrary.a
	nim c --passL:clibrary.a --passL:-llapacke main.nim

