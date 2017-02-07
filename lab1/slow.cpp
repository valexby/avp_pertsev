#include <cstdlib>
#include <iostream>
#include <ctime>
#include <xmmintrin.h>
#include <cstring>

using namespace std;

const size_t GLOB_SZ = 128, ELEM_SZ = 4, VECT_SZ = 4;

union Mat {
	float m[ELEM_SZ][ELEM_SZ];
	__m128 row[ELEM_SZ][ELEM_SZ/VECT_SZ];
};

void gen_element(Mat &element)
{
	int ran1, ran2;
	size_t i, j;
	srand(clock());
	ran1 = rand();
	for (i=0;i<ELEM_SZ;i++)
	{
		for(j=0;j<ELEM_SZ;j++)
		{
			ran2 = rand();
			element.m[i][j] = (float)ran1/(float)ran2;
		}
	}
}

template <size_t N>
static inline void lines_sum(__m128 (&out) [N], const __m128 *addition)
{
	size_t i;
	for (i=0;i<ELEM_SZ/VECT_SZ;i++)
		out[i] = _mm_add_ps(out[i], addition[i]);
}

template <size_t N>
static inline __m128* vector_mul_line(const __m128 &a, const __m128 (&b) [N], int pos)
{
	size_t i;
	__m128 multiplier, *result = new __m128[ELEM_SZ/VECT_SZ];
	switch (pos)
	{
		case 0:
			multiplier = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0));
			break;
		case 1:
			multiplier = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1));
			break;
		case 2:
			multiplier = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2));
			break;
		case 3:
			multiplier = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3));
			break;
	}
	for (i=0;i<ELEM_SZ/VECT_SZ;i++) 
		result[i] = _mm_mul_ps(multiplier, b[i]);
	return result;
}

template <size_t N>
static inline void lincomb_SSE(__m128 (&out) [N], const __m128 (&a) [N], const Mat &B)
{
	size_t i;
	for (i=0;i<ELEM_SZ;i++)
	{
		__m128 *mul_res = vector_mul_line(a[i/VECT_SZ], B.row[i], i % VECT_SZ);
		if (i == 0)
			memcpy(out, mul_res, sizeof(out));
		else
			lines_sum(out, mul_res);
		delete [] mul_res;
	}
}

void matmult_SSE(Mat &out, const Mat &A, const Mat &B)
{
	size_t i;
	for (i=0;i<ELEM_SZ;i++)
	{
		 lincomb_SSE(out.row[i], A.row[i], B);
	}
}

void matmult_simple(Mat &out, const Mat &A, const Mat &B)
{
	size_t i, j;
	for (i=0;i<ELEM_SZ;i++)
		for (j=0;j<ELEM_SZ;j++)
		{
			out.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j] + A.m[i][3] * B.m[3][j];
		}
}


void matadd(Mat &out, const Mat &A)
{
	size_t i, j;
	for (i=0;i<ELEM_SZ;i++)
		for (j=0;j<ELEM_SZ;j++)
			out.m[i][j] += A.m[i][j];
}

float run_test(void (*multiply)(Mat&, const Mat &, const Mat &), 
		Mat **out, Mat **A, Mat **B)
{
	size_t i, j, k;
	clock_t begin = clock();
	for (i=0;i<GLOB_SZ;i++)
	{
		for (j=0;j<GLOB_SZ;j++) {
			for (k=0;k<GLOB_SZ;k++)
				if (k==0)
					multiply(out[i][j], A[i][k], B[k][j]);
				else
				{
					Mat buff;
					multiply(buff, A[i][k], B[k][j]);
					matadd(out[i][j], buff);
				}
		}
	}
	return clock() - begin;
}

bool matcmp(Mat **A, Mat **B)
{
	size_t i, j;
	for (i=0;i<GLOB_SZ;i++)
		for (j=0;j<GLOB_SZ;j++)
			if (memcmp(A[i][j].m, B[i][j].m, sizeof(A[i][j])) != 0)
				return false;
	return true;
}

int main()
{
	size_t i, j;
	union Mat **m1 = new Mat*[GLOB_SZ], **m2 = new Mat*[GLOB_SZ],
	      **res_simple = new Mat*[GLOB_SZ], **res_sse = new Mat*[GLOB_SZ];
	for (i=0;i<GLOB_SZ;i++)
	{
		m1[i] = new Mat[GLOB_SZ]; 
		m2[i] = new Mat[GLOB_SZ];
		res_simple[i] = new Mat[GLOB_SZ];
	       	res_sse[i] = new Mat[GLOB_SZ];
		for(j=0;j<GLOB_SZ;j++)
		{
			gen_element(m1[i][j]);
			gen_element(m2[i][j]);
		}
	}
	clock_t time_simple = run_test(&matmult_simple, res_simple, m1, m2);
	clock_t time_sse = run_test(&matmult_SSE, res_sse, m1, m2);
	if ( !matcmp(res_simple, res_sse) )
		cout << "Test failed!\n";
	else
		cout << "C time: " << (float)time_simple / CLOCKS_PER_SEC << 
			"\nSSE time: " << (float)time_sse / CLOCKS_PER_SEC << endl;
	for (i=0;i<GLOB_SZ;i++)
	{
		delete[] m1[i];
		delete[] m2[i];
		delete[] res_simple[i];
		delete[] res_sse[i];
	}
	delete[] m1;
	delete[] m2;
	delete[] res_simple;
	delete[] res_sse;

	return 0;
}
