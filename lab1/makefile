debug:
	g++ -fno-builtin -g fast.cpp

all: slow.o fast.o
	g++ -o fast fast.o
	g++ -o slow slow.o

slow.o: slow.s
	g++ -c slow.s

fast.o: fast.s
	g++ -c fast.s

fast.s: fast.cpp
	g++ -S -Wall -Werror -Wpedantic fast.cpp

slow.s: slow.cpp
	g++ -S -O0 -Wall -Werror -Wpedantic slow.cpp

clean:
	rm *.o fast slow *.s
