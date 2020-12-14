#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#define  ull long long int

size_t min(size_t m, size_t n){
	if( m<=n)	
		return m;
	else return n;
}
double mean(double *a, ull n){
        double m = 0.0;
        for(ull i = 0; i<n; i++)
                m += a[i];
        return m / (double)n;
}

double** set_input_matrix(const int m, const int n){
        double** a = malloc(sizeof(double*) * m);
        for(int i=0; i<m; i++)
                a[i] = malloc(sizeof(double) * n);

        for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                        a[i][j] = (rand() % RAND_MAX)/10e8;
                }
        }
        return a;
}

double* mat_vect(double** A, double** v, const int m, const int n, const int k){
        double* col_tmp = calloc(m, sizeof(double));
        for(int i=0; i<m; i++)
                for(int j=0; j<n; j++){
                        col_tmp[i] += A[i][j]*v[j][k];
                }
        for(int j=0; j<m; j++)
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

void QR_Factorization( double* const Q, double* const R, double* A, const size_t m, const size_t n) {
	double* tmp = malloc(sizeof(double) * m * n);
    	const size_t rank = min(m, n);
   	
	if(m == n) {
        	memcpy(tmp, A, sizeof(double)*(m * n));
    	} else {
        	for(int i = 0; i < m; i++) {
            	memcpy(tmp + i * rank, A + i * n, sizeof(double)*(rank));
        	}
    	}
       	double* tau = (double*)malloc(sizeof(double) * rank);

    	// Compute QR factorisations
    	LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int) m, (int) n, A, (int) n, tau);
	
  	
	// Copy the corrisponding part of R upper triagular matrix
    	for(int i =0;  i < rank; i++) {
        	memset(R + i * n, 0, i*sizeof(double));
        	memcpy(R + i * n + i, A + i * n + i, (n - i) * sizeof(double));
    	}

    	// Compute orthogonal matrix Q 
    	LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int) m, (int) rank, (int) rank, A, (int) n, tau);

	if(m == n) {
        	memcpy(Q, A, sizeof(double)*(m * n));
    	} else {
        	for(int i =0; i < m; i++) {
            	memcpy(Q + i * rank, A + i * n, sizeof(double)*(rank));
        	}
    	}

    	if(m == n) {
        	memcpy(A, tmp, sizeof(double)*(m * n));
    	} else {
        	for(int i =0; i < m; i++) {
            	memcpy(A + i * rank, tmp + i *n, sizeof(double)*(rank));
        	}
    	}
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
                for(int i=0; i<m; i++){
                        double tmp_u = (*u)[i][k];
                        assert((*alpha)[k] != 0);
                        (*u)[i][k] = tmp_u / (*alpha)[k];
                }

                // vk+1 = A* uk - alphak vk
                rec_vk1(k, A, m, n, v, *u, *alpha);
                (*beta)[k] = norm_2(*v, k+1, n);
                for(int i=0; i<n; i++){
                        double tmp_u = (*v)[i][k+1];
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
        (*diag_t)[0] =  alp * alp;
        for(int i=1; i<n; i++){
                alp = (*alpha)[i];
                alp_= (*alpha)[i-1];

                bet = (*beta)[i-1] ;

                (*diag_t)[i] = (alp * alp) + (bet * bet);
                (*sdiag_t)[i-1]= alp_ * bet;
        }
}

void normalize(const int n, double* u){
        double norm = 0;
        for(int i = 0; i < n; i++){
                ull uu = u[i];
                norm += uu*uu;
        }
        assert(norm > 0);
        norm = sqrt(norm);
        assert(norm != 0);
        for(int i = 0; i < n; i++)
                u[i]=u[i] / norm;
}

void set_QR_parameters(double** H, double epsilon, double** UT, double** VT, double* diag_t, double* sdiag_t, int n){
		
	for(int i = 0; i<n ; i++){
		(*H)[i + i*n] = diag_t[i];
		if(i != 0)
			(*H)[(i-1) + i * n] = sdiag_t[i-1];	
	
		if(i != n-1)
			(*H)[(i+1) + i * n] = sdiag_t[i];
		(*UT)[i + i * n] = 1.;
		(*VT)[i + i * n] = 1.;
	}
}

int Convergence(double** Hk, double** Hk_, int n, double epsilon){
/*	puts("Hk_");
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			printf("%lf\t", (*Hk_)[j  + i * n]);
		puts("\n");
	}
	puts("Hk");
	for(int i = 0; i<n; i++){
			for(int j = 0; j<n; j++)
				printf("%lf\t", (*Hk)[j  + i * n]);
			puts("\n");
	}
*/
	int test = 0;
	double l = 1., ll = 0., s = 1., ss = 0.;
	for(int i=0; i<n; i++){
		if (i != 0){
			l  = (*Hk)[(i-1) + i * n] ;
			ll = (*Hk_)[(i-1) + i * n];
			assert(ll != 0);
		}
		if (i != n-1){
			s  = (*Hk)[(i+1) + i * n] ;
			ss = (*Hk_)[(i+1) + i * n];
			assert(ss != 0);
		}
			
		if (ll / l <= epsilon)
			test ++ ;
		if (ss / s <= epsilon)
			test ++ ;
		//printf("test : %d\n", test);
	}
	return test;
}

void update_matrix(double** H, double** Q, double** R, double u, int n){
	double tmp = 0.;
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			for(int k = 0; k<n; k++){
				tmp += (*R)[k + i * n] * (*Q)[i + k * n];
				if (i == j)
					tmp += u;
				(*H)[j + i * n] = tmp;			
			}		
		}
	}
}	


void left_right_vect(double* UT, double* VT, double* R, double* Q, int n){
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			for(int k = 0; k<n; k++){
				VT[i + j * n] = VT[k + i * n] *  R[i + k * n];
				UT[i + j * n] =  Q[i + k * n] * UT[i + k * n];
			}
	}
}

void backward_mat(double** Ua, double* UT, double* U, const int mu, const int nu, double* z, double* VT, double* V, const int nv){
		for(int i = 0; i<mu; i++)
			for(int j = 0; j<nu; j++)
				for(int k = 0; k<nu; k++)
					U[j + i * nu] = Ua[i][k] * UT[i + k * nu];
		
		for(int i = 0; i<nv; i++)
			for(int j = 0; j<nv; j++)
				for(int k = 0; k<nv; k++)
					V[j + i * nv] = z[i + k * nv]* VT[i + k * nv];	
}

int main(int argc, char** argv){
	srand(time(NULL));
        double** A;

	if(argc != 3){
                perror("Wrong arguments !\n");
                exit(0);
        }

        int m = atoi(argv[1]), n = atoi(argv[2]);

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


	double* a = calloc( n * n, sizeof(double));

        //2.2 using tridiagonalization reduction -------------------------------------------------------------------------------------------------

        // Compute t(B) * B = Tridiagonal matrix T = V*Sigma^2*VT  -----------------------

        double* diag_t = malloc(sizeof(double) * n);
        double* sdiag_t= malloc(sizeof(double) * (n-1));
        normal_product(n, &alpha, &beta, &diag_t, &sdiag_t);

	/* Compute eigenvalues decompsition of the Tridiagonal matrix using Shifted QR method */
	
	double* Q = malloc(sizeof(double) * n*n);
	double* R = malloc(sizeof(double) * n*n);

	double* H = calloc(n * n, sizeof(double));

	double* UT = calloc(n * n, sizeof(double));
	double* VT = calloc(n * n, sizeof(double));

	double epsilon = 10e-10;

	set_QR_parameters(&H, epsilon, &UT, &VT, diag_t, sdiag_t, n);

	double* HH = calloc(n * n, sizeof(double));
	for(int i = 0; i<n; i++){
		HH[i + i * n] = 1.;
		if(i != 0)
			HH[(i-1) + i * n] = 1.;
		if(i != n - 1)
			HH[(i+1) + i * n] = 1.;
		VT[i + i * n] = 1.;
		UT[i + i * n] = 1.;	
	}
	double uu = 0. ; // shift
	double* tmp_ = malloc(sizeof(double) * n * n);
	int k = 0;

	/* Shifted QR method */
	do{

		uu = H[n-1 + (n-1) * n]; //shift  Tnn	
		
		QR_Factorization(Q, R, H, n, n); // qr(Hk - µk*I)
		tmp_ = H; 
		update_matrix(&H, &Q, &R, uu, n);    // Hk+1 = Rk+1 Qk+1 + µk*I			
		left_right_vect(UT, VT, R, Q, n);		
		HH = H;
		k++;		
	}while( !Convergence(&tmp_, &HH, n, epsilon) );

	// we got UT and VT : left and right eigenvalues of T
	// and H tends to a diagonal matrix : eigenvalues! ==> sigular values !

	/* final results ---*/	
	// Sigma = H 
	// U    =  u * UT
	// t(V) =  t(z) * VT
	
	double* U = malloc(sizeof(double) * m * n);
	double* V = malloc(sizeof(double) * n * n);
	
	double** Ua = malloc(
			sizeof(double*) * m);
	for(int i = 0; i<m; i++)
		Ua[i] = malloc(sizeof(double) * n);
	
	double* z = malloc(sizeof(double) * n * n);
	
	backward_mat(Ua, UT, U, m, n, z, VT, V, n);
	

}
