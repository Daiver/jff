.PHONY: all clean .FORCE

main: .FORCE runtests
	ghc -o main main.hs
	./runtests

runtests:
	ghc -o runtests runTests.hs

clean:
	-rm main runtests *.o *.hi

.FORCE:
	make clean
