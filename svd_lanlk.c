#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#define uli unsigned long int

size_t min(size_t m, size_t n){
        if( m<=n)
                return m;
        else return n;
}

double** set_input_matrix(const int m, const int n){
	double** a = malloc(sizeof(double*) * m);
	for(int i=0; i<m; i++)
		a[i] = malloc(sizeof(double) * n);

	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			a[i][j] = (rand() % RAND_MAX)/10e5;
			//printf("%lf\t", a[i][j]);
		}
		//printf("\n");
	}	
	return a;
}


void print_matrix(const int m, const int n, double** A){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			printf("%lf\t", A[i][j]);
		printf("\n");
	}
}


void normalize(const int n, double* u){
	double norm = 0;
	for(int i = 0; i < n; i++){
		uli uu = u[i];
		norm += uu*uu;
	}
	assert(norm > 0);
	norm = sqrt(norm);
	assert(norm != 0);
	for(int i = 0; i < n; i++)
		u[i]=u[i] / norm;
}


double* mat_vect(double** A, double** v, const int m, const int n, const int k){
	double* col_tmp = calloc(m, sizeof(double));
	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++){
			printf("A[%d][%d] * v[%d][%d] = %lf * %lf\n", i, j, j, k, A[i][j], v[j][k]);	
			col_tmp[i] += A[i][j]*v[j][k];
		}
	for(int j=0; j<m; j++)
		printf("col_tmp[%d] = %lf\n", j, col_tmp[j]);
	return col_tmp;
}


double* mat_vect_t(double** A, double** u, const int m, const int n, const int k){
	double* col_tmp = calloc(n, sizeof(double));
	for(int i=0; i<n; i++)
		for(int j=0; j<m; j++)
			col_tmp[i] += A[j][i]*u[i][k];
	return col_tmp;
}


double norm_2(double** u, const int k, const int m){
	double norm = 0.;
	for(int i=0; i<m; i++){
		double tmp = u[i][k];
		norm += tmp*tmp;
	}
	//printf("%lf\n", norm);
	//assert(norm > 0);
	norm = sqrt(norm);
	return norm;
}


void rec_uk(const int k, double** A, const int m, const int n ,double** v, double*** u, double* beta){
	double* col_tmp = malloc(sizeof(double) * m);
	col_tmp = mat_vect(A, v, m, n, k);	
	for(int j=0; j<m; j++)
		printf("col_tmp[%d] = %lf\n", j, col_tmp[j]);
	for(int i=0; i<m; i++){
		if(k == 0)
			(*u)[i][k] = col_tmp[i];
		else
			(*u)[i][k] = col_tmp[i] - beta[k-1] * (*u)[i][k-1];
	}
}


void rec_vk1(const int k, double** A, const int m, const int n, double*** v, double** u, double* alpha){
	double* col_tmp = malloc(sizeof(double) * n);
	col_tmp = mat_vect_t(A, u, m, n, k);
	for(int i=0; i<n; i++)
		(*v)[i][k+1] = col_tmp[i] - alpha[k] * (*v)[i][k];
}


void Golub_Kahan_Lanczos(double** A, const int m, const int n, double** alpha, double** beta, double*** u, double*** v){
	// Output : B, U, V : U*AV=B bidiagonal
	
	// v1 unit1 - 2norm
	for(int i=0; i<n; i++){
		if (i == 0)
			(*v)[i][0] = 1;
		else
			(*v)[i][0] = 0;
	}
	
	// beta0 = 0

	for(int k = 0; k < n; k++){
		// uk = Av_k - beta_k-1 u_k-1
		rec_uk(k, A, m, n, *v, u, *beta);
		//for(int j=0; j<m; j++)
			//printf("u[%d][%d] = %lf\n", j,k, u[j][k]);
		(*alpha)[k] = norm_2(*u, k, m);
		printf("iteration %d\n", k);
		for(int i=0; i<m; i++){
			double tmp_u = (*u)[i][k];
			printf("alpha : i = %d\n", i);
			assert((*alpha)[k] != 0);
			(*u)[i][k] = tmp_u / (*alpha)[k];
		}
		
		// vk+1 = A* uk - alphak vk
		rec_vk1(k, A, m, n, v, *u, *alpha);
		(*beta)[k] = norm_2(*v, k+1, n);
		for(int i=0; i<n; i++){
			double tmp_u = (*v)[i][k+1];
			printf("beta : i = %d\n", i);
			assert((*beta)[k] != 0);
			(*v)[i][k+1] = tmp_u / (*beta)[k];
		}
	}
	// At the end we have B = bidiagonal[diagonal : alpha, superdiagonal : beta]
	// And U = u  = [u1, u2, ... un], V = v = [v1, v2, ... vn]
	// Must add U, V, B in the parameters to be returned of the function !
}


void normal_product(const int n, double** alpha, double** beta, double** diag_t, double** sdiag_t){
	double alp = (*alpha)[0];
	double bet = 0., alp_ = 0.;
	(*diag_t)[0] = 	alp * alp;	
	for(int i=1; i<n; i++){
		alp = (*alpha)[i];
		alp_= (*alpha)[i-1];

		bet = (*beta)[i-1] ;

		(*diag_t)[i] = (alp * alp) + (bet * bet);
		(*sdiag_t)[i-1]= alp_ * bet;
	}
}


void prod_mat_mat(double **U, double** Ua, const int m, const int n, double** Ub, const int mm, const int nn){
 // matrix-matrix product : 2d array
	for(int i=0; i<m; i++){
       		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++)
				U[i][j] = Ua[i][k] * Ub[k][i];
		}
	}	
}


void prod_mat_mat_array(double** V, double* z, const int n, const int nn, double** v, const int m, const int mm){
 // matrix-matrix product : 1d array : compute : t(z) * t(v)
 	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
				V[i][j] = z[i + j * n] * v[i][j];
		}
	}
}

void QR_Factorization( double* const _Q, double* const _R, double* const _A, const size_t _m, const size_t _n) {
    // Maximal rank is used by Lapacke
    //const size_t rank = std::min(_m, _n); 

	const size_t rank = min(_m, _n);

    // Tmp Array for Lapacke
    //const std::unique_ptr<double[]> tau(new double[rank]);
	
	double* tau = (double*)malloc(sizeof(double) * rank);

    // Calculate QR factorisations
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int) _m, (int) _n, _A, (int) _n, tau);

    // Copy the upper triangular Matrix R (rank x _n) into position
    for(size_t row =0; row < rank; ++row) {
        memset(_R+row*_n, 0, row*sizeof(double)); // Set starting zeros
        memcpy(_R+row*_n+row, _A+row*_n+row, (_n-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
    }

    // Create orthogonal matrix Q (in tmpA)
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int) _m, (int) rank, (int) rank, _A, (int) _n, tau);

    //Copy Q (_m x rank) into position
    if(_m == _n) {
        memcpy(_Q, _A, sizeof(double)*(_m*_n));
    } else {
        for(size_t row =0; row < _m; ++row) {
            memcpy(_Q+row*rank, _A+row*_n, sizeof(double)*(rank));
        }
    }
}
	


int main(int argc, char** argv){
	srand(time(NULL));
	if(argc != 3){
		perror("Wrong arguments !\n");
		exit(0);
	}

	int m = atoi(argv[1]), n = atoi(argv[2]);
	double** A;
	A = set_input_matrix(m, n); //set random input matrix A

	//1. Reduction to bidiagonal form B--------------------------------------------------------------------------------------------------------
	
	double** v = malloc(sizeof(double*) * (n+1));// un colonne de plus pour éviter le débordement dans la fonction de récurrence rec_vk!
	double** u = malloc(sizeof(double*) * m);
	for(int i=0; i<n+1; i++)
		v[i] = malloc(sizeof(double*) * (n+1));
	for(int i=0; i<m; i++)
		u[i] = malloc(sizeof(double*) * n);

	double* alpha = malloc(sizeof(double) * n);      // Diagonal elements
	double* beta  = malloc(sizeof(double) * (n - 1));// superdiagonal elements
	
	Golub_Kahan_Lanczos(A, m , n,  &alpha, &beta, &u, &v);
	
	printf("digonal elements : \n");
	for(int i=0; i<n; i++){
		printf("%lf\t", alpha[i]);
	}
	printf("superdigonal elements : \n");
	for(int i=0; i<n-1; i++)
		printf("%lf\t", beta[i]);
	
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++)
			printf("u[%d][%d] = %lf\t", i, j, u[i][j]);
		printf("\n");
	}
	for(int i=0; i<n + 1; i++){
		for(int j=0; j<n; j++)
			printf("v[%d][%d] = %lf\t", i, j, v[i][j]);
		printf("\n");
	}

/*	
	//2. Find sepctral decomposition of the bidiagonal matrix B ----------------------------------------------------------------------------------------
	//First we transform our matrix to a tridiagonal form, then we search for the spectral decomposition of this matrix 
*/	
	// Compute t(B) * B = Tridiagonal matrix T = V*Sigma^2*VT -----------------------

	double* diag_t = malloc(sizeof(double) * n);
	double* sdiag_t= malloc(sizeof(double) * (n-1));
	normal_product(n, &alpha, &beta, &diag_t, &sdiag_t);
	
	printf("associated tridiagonal matrix : \n");
	for(int i=0; i<n; i++)
		printf("%lf\t", diag_t[i]);
	printf("symetric elements : \n");
	for(int i=0; i<n-1; i++)
		printf("%lf\t", sdiag_t[i]);
	printf("\n");
	//----------------------------------------------------------------------------
	// use LAPACKE routine 'dstevx' to compute sepctral decomposition/singular of T (Real Symetric tridiagonal matrix)-------------------
	double vl = 0., vu = 0.;
	lapack_int il, iu, mm, ldz, info;
	double abstol = 10e-8;
	// lapack_int* mm = NULL; // the total number of eigenvalues found (= n, range = 'A') // ERROR !!!!
	ldz = n;
	double* w =  malloc(sizeof(double) * n );
	double* z =  malloc(sizeof(double) * n * n);
	
	lapack_int* ifail = (lapack_int*)malloc(sizeof(lapack_int) * n * 2);

	double* d = malloc(sizeof(double) * n);
	double* e = malloc(sizeof(double) * (n-1));
	d = diag_t;
	e = sdiag_t; 

	info = LAPACKE_dstevx(LAPACK_ROW_MAJOR, 'V', 'A', n, d, e, vl, vu, il, iu, abstol, &mm, w, z, ldz, ifail);
	printf("info = %d\n", info);
	
	// Output w : eigenvalues  &  z : eigenvectors of T ------------------------------------------------------------------------------

	// Last step : backward computation of singular vectors : -------------------------------------------------------------------------
	
	// A = Ua * B * t(Va) = Ua * ( Ub * Sigma * t(Vb)) * t(Va) = (Ua * Ub) * Sigma * (t(Vb) * t(Va)) = U * Sigma * t(V)
	// Or t(B) * B = T tridiagonal = Vb * Sigma² * t(Vb)
	// Sigma_i = sqrt(Sigma²_i), with Sigma² = w & Vb = z;
	// We have to compute Ub ! Ub = B * Vb * Sigma⁻¹
	// with B(output of Golub-Kahan-Lanczos), & Vb = z eigenvectors of T = t(B) * B
	// Ua = u & Va = v : output of Golub-Kahan-Lanczos


	double** Ub = malloc(sizeof(double*) * n);
	for(int i=0; i<n; i++)
		Ub[i] = malloc(sizeof(double) * n);

	double* Sigma = malloc(sizeof(double) * n);
	double ww;
	for(int i=0; i<n; i++){
		ww = w[i];
		Sigma[i] = sqrt(ww);
		printf("Sigma_%d = %lf\n", i, Sigma[i]);
	}
	double** B_Vb = malloc(sizeof(double*) * n);
	for(int i=0; i<n; i++)
		B_Vb[i] = malloc(sizeof(double) * n);
	for(int i=0; i<n; i++){
		//printf("i = %d\n", i);
		for(int j=0; j<n; j++){
			//printf("j = %d\n", j);
			if (i == n-1){
				B_Vb[n - 1][j] = alpha[n-1] * z[j + (n - 1) * n];
				break;
			}
			B_Vb[i][j] = alpha[i] * z[j + i * n] + beta[i] * z[(j + i * n) + 1];
		}
	}

	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++){
			Ub[i][j] = B_Vb[i][j] / (Sigma[j]);
		}

	// Compute final sigular vectors :

	double** U = malloc(sizeof(double) * m * n);
	for(int i=0; i<m; i++)
		U[i] = malloc(sizeof(double) * n);
	double** V = malloc(sizeof(double) * n * n);
	for(int j=0; j<n; j++)
		V[j] = malloc(sizeof(double) * n);
	
	// U = Ua * Ub
	prod_mat_mat(U, u, m, n, Ub, n, n);

	// V = t(Vb) * (Va) = t(z) * t(v)
	prod_mat_mat_array(V, z, n, n, v, n, n);

	// Fin de l'algo : A = U * Sigma * t(V) ---------------------------------------------------------------------------------------------


	// Second method : Use shift-QR method to compute eigenvalue decomposition of the tridiagonal matrix T  : Q^TQ = D,
	// benchmarks :  for each implementation, tests the main program a RUN number of times, then measure time, throughput, ..

}	
