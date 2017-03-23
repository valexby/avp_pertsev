all : main.o
	gcc -lOpenCL -o main main.o

main.o : main.c
	gcc -g -Wall -c main.c

clean :
	rm main *.o
