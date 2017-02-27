#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <immintrin.h>

#define size 1024   //1774
#define sizeC 1774

void multiply(float** matrix1, float** matrix2, float** &matrix3) {
	for(int k = 0; k < size; k += 4) {
		for(int i = 0; i < size; i++) {
			for(int j = 0; j < size; j++) {
					matrix3[i][j] += matrix1[i][k] * matrix2[k][j]
					       	+ matrix1[i][k+1] * matrix2[k+1][j]
					       	+ matrix1[i][k+2] * matrix2[k+2][j] 
						+ matrix1[i][k+3] * matrix2[k+3][j];
			}
		}
	}
}

void multiplySSE(float** &matrix1, float** &matrix2, float** &matrix3) {
	__m128 res[size][size/4];

	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			__m128 tmp = _mm_load1_ps(&matrix1[i][j]);
			for(int k = 0; k < size; k += 4) {
				if(j == 0)
					res[i][k/4] = _mm_mul_ps(tmp, _mm_load_ps(&matrix2[j][k]) );
				else
					res[i][k/4] = _mm_add_ps(res[i][k/4], _mm_mul_ps( tmp, _mm_load_ps(&matrix2[j][k]) ) );
			}
		}

		for(int l = 0; l < size; l += 4) {
			_mm_store_ps(&matrix3[i][l], res[i][l/4]);
		}
	}
}

int main() {

	float **matrix1, **matrix2, **matrix3, **matrix4;

	matrix1 = (float**) _mm_malloc(size * sizeof(float*), 16);
	matrix2 = (float**) _mm_malloc(size * sizeof(float*), 16);
	matrix3 = (float**) _mm_malloc(size * sizeof(float*), 16);
	matrix4 = (float**) _mm_malloc(size * sizeof(float*), 16);

	for(int i = 0; i < size; i++) {
		matrix1[i] = (float*) _mm_malloc(size * sizeof(float), 16);
		matrix2[i] = (float*) _mm_malloc(size * sizeof(float), 16);
		matrix3[i] = (float*) _mm_malloc(size * sizeof(float), 16);
		matrix4[i] = (float*) _mm_malloc(size * sizeof(float), 16);
	}

	srand(time(NULL));

	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			matrix1[i][j] = 0 + rand() % 5;
			matrix2[i][j] = 0 + rand() % 5;
			matrix3[i][j] = 0;
		}
	}

	unsigned int startTime, endTime;

	printf("Start\n");

	startTime = clock();
	
	multiply(matrix1, matrix2, matrix3);

	endTime = clock();

	printf("End\n");
	printf("Time(ms):%u\n\n", (endTime - startTime));


	printf("Start SSE\n");

	startTime = clock();

	multiplySSE(matrix1, matrix2, matrix4);

	endTime = clock();
	
	printf("End SSE\n");
	printf("Time SSE(ms):%u\n\n", (endTime - startTime));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			if (matrix3[i][j] != matrix4[i][j])
				printf("Test failed!");
			
	for(int i = 0; i < size; i++) {
		_mm_free(matrix1[i]);
		_mm_free(matrix2[i]);
		_mm_free(matrix3[i]);
		_mm_free(matrix4[i]);
	}

	_mm_free(matrix1);
	_mm_free(matrix2);
	_mm_free(matrix3);
	_mm_free(matrix4);

	return 0;
}
