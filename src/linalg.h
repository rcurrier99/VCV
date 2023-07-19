// Header File with Linear Algebra Structs
// Ry Curier 2023-06-20

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>

#ifndef linalg
#define linalg

#define MAT_AT(M, i, j) ((M).vals[(i)*(M).cols + (j)])
#define TRI_MAT_AT(T, i, j) ((T).vals[((T).cols-1)*((T).cols)/2 - ((T).cols-(j+1)-1)*((T).cols-(j+1))/2 - ((T).cols-1-i) - 1])

template <typename T> struct Mat {
	size_t rows;
	size_t cols;
	T* vals;

	Mat(size_t M, size_t N) : rows(M), cols(N) {
		vals = (T*) malloc(sizeof(T) * M * N);
		assert(vals != NULL);
	}

	void smul(const T& s) {
		for (size_t i=0; i<rows; i++) {
			for (size_t j=0; j<cols; j++) {
				MAT_AT(*this, i, j) *= s;	
			}
		}
	}

	void fill(const T& value) {	
		for (size_t i=0; i<rows; i++) {
			for (size_t j=0; j<cols; j++) {
				MAT_AT(*this, i, j) = value;
			}
		}
	}

	void fill_mat (const Mat<T>& other) {
		assert(rows == other.rows && cols == other.cols);
		for (size_t i=0; i<rows; i++) {
			for (size_t j=0; j<cols; j++) {
				MAT_AT(*this, i, j) = MAT_AT(other, i, j);
			}
		}
	}

	void print() {
		for (size_t i=0; i<rows; i++) {
			for (size_t j=0; j<cols; j++) {
				std::cout << MAT_AT(*this, i, j) << " ";
			}
			std::cout << std::endl;
		}
	}
};

template <typename T> struct TriMat : Mat<T> {
	TriMat(size_t N) : Mat<T>(N, N) {
		this->vals = (T*) malloc(sizeof(T) * (N * (N-1)/2));
	}
};

template <typename T> struct SymToeplitzMat {
	size_t size;
	size_t diags;
	Mat<T> vals;
	Mat<T> D;
	TriMat<T> L;

	SymToeplitzMat(size_t n, size_t d) : size(n), diags(d), vals(d, 1), D(n, 1), L(n) {
	}

	void factor() {
		T sum;
		for (size_t i=0; i<size; i++) {
			sum = 0;
			for (size_t k=0; i-k<diags; k++)
				sum += pow(TRI_MAT_AT(L, i, k), 2) * MAT_AT(D, k, 0);
			MAT_AT(D, i, 0) = MAT_AT(vals, 0, 0) - sum;
			for (size_t j=i+1; j<std::min(size, i+diags); j++) {
				sum = 0;
				for (size_t k=0; i-k<diags; k++)
					sum += TRI_MAT_AT(L, j, k) * TRI_MAT_AT(L, i, k) * MAT_AT(D, k, 0);
				TRI_MAT_AT(L, j, i) = MAT_AT(vals, std::abs((int)i - (int)j), 0) / MAT_AT(D, i, 0);
			}
		}	
	}
};

template <typename T> void mat_dot(Mat<T>& lhs, const Mat<T>& rhs1, const Mat<T>& rhs2) {
	assert(rhs1.cols == rhs2.rows && lhs.rows == rhs1.rows && lhs.cols == rhs2.cols);
	T sum;
	for (size_t i=0; i<lhs.rows; i++) {
		for (size_t j=0; j<lhs.cols; j++) { 
			sum = 0;
			for (size_t k=0; k<rhs1.cols; k++) {
				sum += MAT_AT(rhs1, i, k) * MAT_AT(rhs2, k, j);
			}
			MAT_AT(lhs, i, j) = sum;
		}
	}
}

template <typename T> void mat_sum(Mat<T>& lhs, const Mat<T>& rhs1, const Mat<T>& rhs2) {
	assert(lhs.rows == rhs1.rows && lhs.rows == rhs2.rows && lhs.cols == rhs1.cols && lhs.cols == rhs2.cols);
	for (size_t i=0; i<lhs.rows; i++) {
		for (size_t j=0; j<lhs.cols; j++) {
			MAT_AT(lhs, i, j) = MAT_AT(rhs1, i, j) + MAT_AT(rhs2, i, j);
		}
	}
}

template <typename T> void mat_diff(Mat<T>& lhs, const Mat<T>& rhs1, const Mat<T>& rhs2) {
	assert(lhs.rows == rhs1.rows && lhs.rows == rhs2.rows && lhs.cols == rhs1.cols && lhs.cols == rhs2.cols);
	for (size_t i=0; i<lhs.rows; i++) {
		for (size_t j=0; j<lhs.cols; j++) {
			MAT_AT(lhs, i, j) = MAT_AT(rhs1, i, j) - MAT_AT(rhs2, i, j);
		}
	}
}

template <typename T> void fsubst(Mat<T>& x, const TriMat<T>& A, const Mat<T>& b) {
	assert(x.cols == 1 && b.cols == 1 && x.rows == A.rows && A.cols == b.rows);
	T sum;
	for (size_t i=0; i<A.rows; i++) {
		sum = MAT_AT(b, i, 0);
		for (size_t j=0; j<i; j++) {
			sum -= TRI_MAT_AT(A, i, j) * MAT_AT(x, j, 0);
		}
		MAT_AT(x, i, 0) = sum;
	}
}

template <typename T> void bsubst(Mat<T>& x, const TriMat<T>& A, const Mat<T>& b) {
	assert(x.cols == 1 && b.cols == 1 && x.rows == A.rows && A.cols == b.rows);
	T sum;
	for (size_t i=A.rows-1; i>=0 && i<A.rows; i--) {
		sum = MAT_AT(b, i, 0);
		for (size_t j=A.cols-1; j>i && j<A.cols; j--) {
			sum -= TRI_MAT_AT(A, j, i) * MAT_AT(x, j, 0);
		}
		MAT_AT(x, i, 0) = sum;
	}
}

template <typename T> void stmat_mul(Mat<T>& b, const SymToeplitzMat<T>& A, const Mat<T>& x) {
	assert(x.cols == 1 && b.cols == 1 && x.rows == A.size && A.size == b.rows);
	T sum;
	for (size_t i=0; i<b.rows; i++) {
		sum = 0;
		for (size_t j=(size_t)std::max(0, int(i-A.diags+1)); j<(size_t)std::min((int)A.size, (int)(i+A.diags)); j++) {
			sum += MAT_AT(A.vals, std::abs( (int)i - (int)j), 0) * MAT_AT(x, j, 0);
		}
		MAT_AT(b, i, 0) = sum;
	}
}

template <typename T> void solve(Mat<T>& x, SymToeplitzMat<T>& A, const Mat<T>& b, Mat<T>& temp1, Mat<T>& temp2) {
	assert(x.cols == 1 && b.cols == 1 && x.rows == A.size && A.size == b.rows && temp1.rows == b.rows && temp2.rows == b.rows);
	A.factor();
	for (size_t i=0; i<b.rows; i++)
		MAT_AT(temp1, i, 0) = MAT_AT(b, i, 0) / MAT_AT(A.D, i, 0);
	fsubst(temp2, A.L, temp1);
	bsubst(x, A.L, temp2);
}

#endif
