/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cugar/analysis/diff.h>
#include <cugar/linalg/tensor.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace cugar {

namespace {

bool is_equal(const float x, const float y) { return fabsf(x-y) < 1.0e-6f; }
void test(const char* name, const float val, const float ref)
{ 
	if (!is_equal(val, ref))
		fprintf(stderr,"    error: expected %s = %f, got %f\n", name, ref, val);
}

// maps the unit square to the sphere with a uniform distribution
template <typename scalar_type>
inline CUGAR_HOST_DEVICE Vector<scalar_type,3> uniform_square_to_sphere(const Vector<scalar_type,2>& uv)
{
	const scalar_type cosTheta = uv.y*2.0f - 1.0f;
	const scalar_type sinTheta = sqrt(max(1.0f - cosTheta*cosTheta,scalar_type(0.0f)));
	const scalar_type phi = uv.x * M_TWO_PIf;

	return Vector<scalar_type,3>(
		cos(phi)*sinTheta,
		sin(phi)*sinTheta,
		cosTheta );
}

// maps the unit square to the sphere with a uniform distribution
template <typename scalar_type>
inline CUGAR_HOST_DEVICE Vector<scalar_type,3> du_uniform_square_to_sphere(const Vector<scalar_type,2>& uv)
{
	const scalar_type cosTheta = uv.y*2.0f - 1.0f;
	const scalar_type sinTheta = sqrt(max(1.0f - cosTheta*cosTheta,scalar_type(0.0f)));
	const scalar_type phi = uv.x * M_TWO_PIf;

	// du(cosPhi)sinTheta = sinTheta * -sin(phi)
	// du(sinPhi)sinTheta = sinTheta * cos(phi)

	return Vector<scalar_type,3>(
		-sin(phi)*sinTheta * M_TWO_PIf,
		cos(phi)*sinTheta * M_TWO_PIf,
		0.0f );
}
// maps the unit square to the sphere with a uniform distribution
template <typename scalar_type>
inline CUGAR_HOST_DEVICE Vector<scalar_type,3> dv_uniform_square_to_sphere(const Vector<scalar_type,2>& uv)
{
	const scalar_type cosTheta = uv.y*2.0f - 1.0f;
	const scalar_type sinTheta = sqrt(max(1.0f - cosTheta*cosTheta,scalar_type(0.0f)));
	const scalar_type phi = uv.x * M_TWO_PIf;

	// dv(cosTetha) = 2
	// dv(sinTheta) = du[sqrt(1 - (4*v^2 - 4v + 1))] = (-8v + 4) * 0.5 / sqrt(1 - (4*v^2 - 4v + 1)) = -(4v + 2) / sqrt(1 - (4*v^2 - 4v + 1))
	// dv(cosPhi)sinTheta = cos(phi) * (2 - 4v) / sqrt(1 - 4v*v)
	// du(cosPhi)sinTheta = sinTheta * -sin(phi)

	const float dsinTheta = (2.0f - 4.0f * uv.y) / sinTheta;
	return Vector<scalar_type,3>(
		cos(phi) * dsinTheta,
		sin(phi) * dsinTheta,
		2.0f );
}

} // anonymous namespace


void diff_test()
{
	fprintf(stderr,"diff test... started\n");
	{
		fprintf(stderr,"  test 1\n");

		// define the basic scalar as a float function of 1 variable with 4th order derivatives
		typedef diff_var<float,1,4> scalar_type;

		// we now declare the variable x wrt which we want to differentiate all expressions
		scalar_type x;

		x.u  = 3.0f;	// set x = 3
		x.du = 1.0f;	// set x' = 1

		// compute an expression b = b(x)
		scalar_type b = x*x;

		float J = diff(b);
		float H = diff(diff(b));
		float T3 = diff(diff(diff(b)));
		float T4 = diff(diff(diff(diff(b))));
		fprintf(stderr,"    b       : %f\n", float(b));
		fprintf(stderr,"    db/dx   : %f\n", J);
		fprintf(stderr,"    db/dx^2 : %f\n", H);
		fprintf(stderr,"    db/dx^3 : %f\n", T3);
		fprintf(stderr,"    db/dx^4 : %f\n", T4);

		test("b(x)",    b,  9.0f);
		test("db/dx",   J,  6.0f);
		test("db/dx^2", H,  2.0f);
		test("db/dx^3", T3, 0.0f);
		test("db/dx^4", T4, 0.0f);
	}
	{
		fprintf(stderr,"  test 2\n");

		// define the basic scalar as a float function of 1 variable with 4th order derivatives
		typedef diff_var<float,1,4> scalar_type;

		// we now declare the variable x wrt which we want to differentiate all expressions
		scalar_type x;

		x.u  = 1.0f;	// set x = 1
		x.du = 1.0f;	// set x' = 1

		// compute an expression b = b(x)
		scalar_type b;
		b = x*x*x*x;
		b = b / x;

		float J = diff(b);
		float H = diff(diff(b));
		float T3 = diff(diff(diff(b)));
		float T4 = diff(diff(diff(diff(b))));
		fprintf(stderr,"    b       : %f\n", float(b));
		fprintf(stderr,"    db/dx   : %f\n", J);
		fprintf(stderr,"    db/dx^2 : %f\n", H);
		fprintf(stderr,"    db/dx^3 : %f\n", T3);
		fprintf(stderr,"    db/dx^4 : %f\n", T4);

		test("b(x)",    b,  1.0f);
		test("db/dx",   J,  3.0f);
		test("db/dx^2", H,  6.0f);
		test("db/dx^3", T3, 6.0f);
		test("db/dx^4", T4, 0.0f);
	}
	{
		fprintf(stderr,"  test 3\n");

		// define the basic scalar as a float function of 1 variable with 4th order derivatives
		typedef diff_var<float,1,4> scalar_type;

		// we now declare the variable x wrt which we want to differentiate all expressions
		scalar_type x;

		x.u  = M_PIf*0.5f;	// set x = PI/2
		x.du = 1.0f;		// set x' = 1

		// compute an expression b = b(x)
		scalar_type b = sin(x);

		float J = diff(b);
		float H = diff(diff(b));
		float T3 = diff(diff(diff(b)));
		float T4 = diff(diff(diff(diff(b))));
		fprintf(stderr,"    b       : %f\n", float(b));
		fprintf(stderr,"    db/dx   : %f\n", J);
		fprintf(stderr,"    db/dx^2 : %f\n", H);
		fprintf(stderr,"    db/dx^3 : %f\n", T3);
		fprintf(stderr,"    db/dx^4 : %f\n", T4);

		test("b(x)",    b,   1.0f);
		test("db/dx",   J,   0.0f);
		test("db/dx^2", H,  -1.0f);
		test("db/dx^3", T3,  0.0f);
		test("db/dx^4", T4,  1.0f);
	}
	{
		fprintf(stderr,"  test 4\n");

		// define the basic scalar as a float function of 1 variable with 4th order derivatives
		typedef diff_var<float,1,4> scalar_type;

		// we now declare the variable x wrt which we want to differentiate all expressions
		scalar_type x;

		x.u  = 1.0f;	// set x = 1
		x.du = 1.0f;	// set x' = 1

		// compute an expression b = b(x)
		scalar_type b = expf(-x*x);

		float J = diff(b);
		float H = diff(diff(b));
		float T3 = diff(diff(diff(b)));
		float T4 = diff(diff(diff(diff(b))));

		fprintf(stderr,"    b       : %f\n", float(b));
		fprintf(stderr,"    db/dx   : %f\n", J);
		fprintf(stderr,"    db/dx^2 : %f\n", H);
		fprintf(stderr,"    db/dx^3 : %f\n", T3);
		fprintf(stderr,"    db/dx^4 : %f\n", T4);
	}
	{
		fprintf(stderr,"  test 5\n");

		// define the basic scalar as a float function of 1 variable with 4th order derivatives
		typedef diff_var<float,1,4> scalar_type;

		// we now declare the variable x wrt which we want to differentiate all expressions
		scalar_type x;

		x.u  = 1.0f;	// set x = 1
		x.du = 1.0f;	// set x' = 1

		// compute an expression b = b(x)
		scalar_type b = 1.0f / (x*x);

		float J = diff(b);
		float H = diff(diff(b));
		float T3 = diff(diff(diff(b)));
		float T4 = diff(diff(diff(diff(b))));
		fprintf(stderr,"    b       : %f\n", float(b));
		fprintf(stderr,"    db/dx   : %f\n", J);
		fprintf(stderr,"    db/dx^2 : %f\n", H);
		fprintf(stderr,"    db/dx^3 : %f\n", T3);
		fprintf(stderr,"    db/dx^4 : %f\n", T4);

		test("b(x)",    b,    1.0f);
		test("db/dx",   J,   -2.0f);
		test("db/dx^2", H,    6.0f);
		test("db/dx^3", T3, -24.0f);
		test("db/dx^4", T4, 120.0f);
	}
	{
		fprintf(stderr,"  test 6\n");

		// define the basic scalar as a float function of 2 variables with 2nd order derivatives
		typedef diff_var<float,2,2> scalar_type;

		// we now declare two variables x and y wrt which we want to differentiate all expressions
		scalar_type x;
		scalar_type y;

		x.u = 2.0f;
		y.u = 2.0f;

		set_primary( 0, x ); // we declare x as the first primary variable to differentiate agaisnt (equivalent to setting dx/dx = 1)
		set_primary( 1, y ); // we declare y as the second primary variable to differentiate agaisnt (equivalent to setting dy/dy = 1)

		// compute an expression b = b(x,y)
		scalar_type b;
		b = x * y;
		b = b + y;

		// and takes its differentials
		Vector2f J = jacobian(b);
		Matrix2x2f H = hessian(b);
		fprintf(stderr,"    db/d(x,y)   : (%f, %f)\n", J[0], J[1]);
		fprintf(stderr,"    db/d(x,y)^2 : (%f, %f)(%f, %f)\n", H(0,0), H(0,1), H(1,0), H(1,1));

		test("db/dx", J[0], 2.0f);
		test("db/dy", J[1], 3.0f);
		test("db/dxx", H(0,0), 0.0f);
		test("db/dxy", H(0,1), 1.0f);
		test("db/dyx", H(1,0), 1.0f);
		test("db/dyy", H(1,1), 0.0f);
	}
	{
		fprintf(stderr,"  test 7\n");

		// define the basic scalar as a float function of 2 variables with 3rd order derivatives
		typedef diff_var<float,2,3> scalar_type;

		// we now declare two variables x and y wrt which we want to differentiate all expressions
		scalar_type x;
		scalar_type y;

		x.u = 1.0f;			// set x = 1
		y.u = 1.0f;			// set y = 1
		x.du[0] = 1.0f;		// we set the derivative of x wrt the first variable (i.e. x) to be 1		(i.e. dx/dx = 1)
		y.du[1] = 1.0f;		// we set the derivative of y wrt the second variable (i.e. y) to be 1		(i.e. dy/dy = 1)

		scalar_type b;
		b = (x * x) * y;
		{
			Vector2f J = jacobian(b);
			Matrix2x2f H = hessian(b);
			Tensor<float,3,2> T3 = diff_tensor<3>(b);
			fprintf(stderr,"    db/d(x,y)   : (%f, %f)\n", J[0], J[1]);
			fprintf(stderr,"    db/d(x,y)^2 : (%f, %f)(%f, %f)\n", H(0,0), H(0,1), H(1,0), H(1,1));
			fprintf(stderr,"    db/d(x,y)^3 :\n");
			fprintf(stderr,"                  (%f, %f)(%f, %f)\n", T3(0,0,0), T3(0,0,1), T3(0,1,0), T3(0,1,1));
			fprintf(stderr,"                  (%f, %f)(%f, %f)\n", T3(1,0,0), T3(1,0,1), T3(1,1,0), T3(1,1,1));

			test("db/dx", J[0], 2.0f);
			test("db/dy", J[1], 1.0f);
			test("db/dxx", H(0,0), 2.0f);
			test("db/dxy", H(0,1), 2.0f);
			test("db/dyx", H(1,0), 2.0f);
			test("db/dyy", H(1,1), 0.0f);

			test("db/dxx", T3(0,0,0), 0.0f);
			test("db/dxy", T3(0,0,1), 2.0f);
			test("db/dyx", T3(0,1,0), 2.0f);
			test("db/dyy", T3(0,1,1), 0.0f);
			test("db/dxx", T3(1,0,0), 2.0f);
			test("db/dxy", T3(1,0,1), 0.0f);
			test("db/dyx", T3(1,1,0), 0.0f);
			test("db/dyy", T3(1,1,1), 0.0f);
		}
		b = b / y;
		{
			Vector2f J = jacobian(b);
			Matrix2x2f H = hessian(b);
			Tensor<float,3,2> T3 = diff_tensor<3>(b);
			fprintf(stderr,"    db/d(x,y)   : (%f, %f)\n", J[0], J[1]);
			fprintf(stderr,"    db/d(x,y)^2 : (%f, %f)(%f, %f)\n", H(0,0), H(0,1), H(1,0), H(1,1));
			fprintf(stderr,"    db/d(x,y)^3 :\n");
			fprintf(stderr,"                  (%f, %f)(%f, %f)\n", T3(0,0,0), T3(0,0,1), T3(0,1,0), T3(0,1,1));
			fprintf(stderr,"                  (%f, %f)(%f, %f)\n", T3(1,0,0), T3(1,0,1), T3(1,1,0), T3(1,1,1));

			test("db/dx", J[0], 2.0f);
			test("db/dy", J[1], 0.0f);
			test("db/dxx", H(0,0), 2.0f);
			test("db/dxy", H(0,1), 0.0f);
			test("db/dyx", H(1,0), 0.0f);
			test("db/dyy", H(1,1), 0.0f);

			test("db/dxx", T3(0,0,0), 0.0f);
			test("db/dxy", T3(0,0,1), 0.0f);
			test("db/dyx", T3(0,1,0), 0.0f);
			test("db/dyy", T3(0,1,1), 0.0f);
			test("db/dxx", T3(1,0,0), 0.0f);
			test("db/dxy", T3(1,0,1), 0.0f);
			test("db/dyx", T3(1,1,0), 0.0f);
			test("db/dyy", T3(1,1,1), 0.0f);
		}
	}
	{
		fprintf(stderr,"  test 8\n");

		// define the basic scalar as a double function of 2 variables with 2nd order derivatives
		typedef diff_var<double,2,2> scalar_type;

		scalar_type x;
		scalar_type y;

		x.u = 2.0;
		y.u = 2.0;
		x.du[0] = 1.0;		// set x' = 1
		y.du[1] = 1.0;		// set y' = 1

		scalar_type b;
		b = sqrt( x * y );

		//d/dx [ sqrt( x * y ) ] = 1/2 y / sqrt( x * y ) = 0.5 * 2 / sqrt(4) = 1/2
		//d/dy [ sqrt( x * y ) ] = 1/2 x / sqrt( x * y ) = 0.5 * 2 / sqrt(4) = 1/2

		Vector2d J = jacobian(b);
		Matrix2x2d H = hessian(b);
		fprintf(stderr,"    db/d(x,y)   : (%f, %f)\n", J[0], J[1]);
		fprintf(stderr,"    db/d(x,y)^2 : (%f, %f)(%f, %f)\n", H(0,0), H(0,1), H(1,0), H(1,1));

		test("db/dx",  (float)J[0], 0.5f);
		test("db/dy",  (float)J[1], 0.5f);
		test("db/dxx", (float)H(0,0), -1.f/8.f);
		test("db/dxy", (float)H(0,1),  1.f/8.f);
		test("db/dyx", (float)H(1,0),  1.f/8.f);
		test("db/dyy", (float)H(1,1), -1.f/8.f);
	}
	{
		fprintf(stderr,"  test 9\n");

		typedef diff_var<float,2,2> scalar_type;

		typedef Vector<scalar_type,2> vec2;
		typedef Vector<scalar_type,3> vec3;

		scalar_type u(0.25f);
		scalar_type v(0.05f);

		set_primary( 0, u );	// set u' = 1
		set_primary( 1, v );	// set v' = 1

		// define a vector p = p(u,v): R^2 -> R^3
		vec3 p = uniform_square_to_sphere( vec2(u,v) );

		// and remember its Jacobian is a 3x2 matrix with each row equal to the gradient of the corresponding component of p
		Matrix3x2f J;
		J[0] = gradient(p.x);	// dx/d(u,v) = grad_uv(x)
		J[1] = gradient(p.y);	// dy/d(u,v) = grad_uv(y)
		J[2] = gradient(p.z);	// dz/d(u,v) = grad_uv(z)

		Vector3f dp_du( J(0,0), J(1,0), J(2,0) );	// dp/du is the first column of J
		Vector3f dp_dv( J(0,1), J(1,1), J(2,1) );	// dp/dv is the second column of J
		Vector3f N = cross( dp_du, dp_dv );

		Vector3f dp_du_ref = du_uniform_square_to_sphere( Vector2f(float(u),float(v)) );
		Vector3f dp_dv_ref = dv_uniform_square_to_sphere( Vector2f(float(u),float(v)) );
		Vector3f N_ref = cross( dp_du_ref, dp_dv_ref );

		fprintf(stderr,"    dp/du       : (%f, %f, %f)\n", dp_du.x, dp_du.y, dp_du.z);
		fprintf(stderr,"    dp/dv       : (%f, %f, %f)\n", dp_dv.x, dp_dv.y, dp_dv.z);
		fprintf(stderr,"    |N|         : %f (%f)\n", length(N), length(N_ref));

		test("dp/du.x",  dp_du.x, dp_du_ref.x);
		test("dp/du.y",  dp_du.y, dp_du_ref.y);
		test("dp/du.z",  dp_du.z, dp_du_ref.z);
		test("dp/dv.x",  dp_dv.x, dp_dv_ref.x);
		test("dp/dv.y",  dp_dv.y, dp_dv_ref.y);
		test("dp/dv.z",  dp_dv.z, dp_dv_ref.z);
	}
	{
		fprintf(stderr,"  test 10\n");

		//
		// This test extends the one above showing how we can also manipulate
		// gradients as differentiable quantities.
		// Hence, we can track the tangents of a mapping from the unit square
		// to the 3-dimensional sphere, compute the normal, and still compute its
		// gradient!
		//

		typedef diff_var<float,2,2> scalar_type;

		typedef Vector<scalar_type,2> vec2;
		typedef Vector<scalar_type,3> vec3;
		typedef Matrix<scalar_type,3,2> matrix3x2;

		scalar_type u(0.25f);
		scalar_type v(0.05f);

		set_primary( 0, u );	// set u' = 1
		set_primary( 1, v );	// set v' = 1

		// define a vector p = p(u,v): R^2 -> R^3
		vec3 p = uniform_square_to_sphere( vec2(u,v) );

		// and remember its Jacobian is a 3x2 matrix with each row equal to the gradient of the corresponding component of p
		matrix3x2 J;
		J[0] = vec2( raise_order(diff(p.x)) );		// grad(p.x) - note we need to raise the differential order to be able to use the original vec2 type (which is twice differentiable)
		J[1] = vec2( raise_order(diff(p.y)) );		// grad(p.y) -    "
		J[2] = vec2( raise_order(diff(p.z)) );		// grad(p.z) -    "

		vec3 dp_du( J(0,0), J(1,0), J(2,0) );	// dp/du is the first column of J
		vec3 dp_dv( J(0,1), J(1,1), J(2,1) );	// dp/dv is the second column of J
		vec3 normal = normalize( cross( dp_du, dp_dv ) );

		// N is still differentiable: we can compute its Jacobian!
		Matrix3x2f JN;
		JN[0] = gradient( normal.x );
		JN[1] = gradient( normal.y );
		JN[2] = gradient( normal.z );

		fprintf(stderr,"    |N|         : %f\n", float(length(normal)));
		fprintf(stderr,"    dN/du       : (%f, %f, %f)\n", JN(0,0), JN(1,0), JN(1,0));
		fprintf(stderr,"    dN/dv       : (%f, %f, %f)\n", JN(0,1), JN(1,1), JN(2,1));

		//
		// and obviously we can also apply a bit of differential geometry to calculate the first and second
		// fundamental forms, and hence, curvature (see https://en.wikipedia.org/wiki/Parametric_surface#Curvature)
		//

		// compute the first fundamental form
		const float E = dot( dp_du, dp_du );
		const float F = dot( dp_du, dp_dv );
		const float G = dot( dp_dv, dp_dv );

		// compute the second-order derivatives of p
		const Vector3f dp_duu(
			gradient(dp_du.x)[0],
			gradient(dp_du.y)[0],
			gradient(dp_du.z)[0] );

		const Vector3f dp_duv(
			gradient(dp_du.x)[1],
			gradient(dp_du.y)[1],
			gradient(dp_du.z)[1] );

		const Vector3f dp_dvv(
			gradient(dp_dv.x)[1],
			gradient(dp_dv.y)[1],
			gradient(dp_dv.z)[1] );

		// compute the second fundamental form
		const float L = dot( dp_duu, Vector3f(normal) );
		const float M = dot( dp_duv, Vector3f(normal) );
		const float N = dot( dp_dvv, Vector3f(normal) );

		// compute Gaussian curvature
		const float K = (L*N - M*M) / (E*G - F*F);
		fprintf(stderr,"    K           : %f\n", K);

		test("K", K, 1.0f); // the Gaussian curvature of a sphere is 1/r^2	- and r = 1 here
	}

	fprintf(stderr,"diff test... done\n");
}

} // namespace cugar
