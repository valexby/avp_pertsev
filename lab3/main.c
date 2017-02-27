#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define SIZE 98304 //3MB / 4B
//#define SIZE 8192*2 //256 KB

int arr[SIZE], dummy[SIZE];

int generate(int *mas)
{
	int i, out = 0;
	srand(clock());
	for (i=0;i<SIZE;i++)
	{
		mas[i] = rand() % 8;
		out += mas[i];
	}
	return out;
}

void clear(void)
{
	int i;
	for (i=0;i<SIZE;i++)
		dummy[i] /= 2;
	for (i=0;i<SIZE;i++)
		dummy[i] *= 2;
}

int main(void)
{
	register size_t i, j, k, d;
	register int sum = 0;
	int check;
	register clock_t time, t_sum;
	scanf("%d", &check);

	generate(&(dummy[0]));

	for (i = 1;i < 2; i++)
	{
		i = 12;
		t_sum = 0;
		for (d = 0;d < 10; d++)
		{
			check = generate(&(arr[0]));
			//clear();
			time = clock();
			sum = 0;
			for (j = 0;j < SIZE / i; j++)
			{
				for (k = j; k < SIZE; k += SIZE / i)
					sum += arr[k];
			}
			t_sum += clock() - time;
			if (check != sum) 
				printf("Error!");
		}
		printf("%ld %ld\n", i, t_sum / 10);
	}
	printf("-1");
	return 0;
}
