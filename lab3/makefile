all: main.o
	gcc -o main main.o

main.o: main.s
	gcc -c main.s

main.s: main.c
	gcc -S -Wall -Werror -Wpedantic -O1 main.c

clean:
	rm *.o main *.s *.png
