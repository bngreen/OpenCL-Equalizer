
#define My_PI 3.1415926535897932384626433832795
#define My_PI2 9.8696044010893586188344909998762

float myAbs(float v)
{
    if(v < 0)
       return -1.0*v;
    else
       return v;
}

float lanczos(float x, int a)
{
	if(x == 0)
		return 1;
	float absX = myAbs(x);
	if(0 < absX && absX < a)
		return (a*sin(My_PI*x)*sin(My_PI*x/a))/(My_PI2*x*x);
	return 0;
}


__kernel void upSample(int by, int N, __global float* input, __global float* output)
{
	int idx = get_global_id(0);
	int center = idx / by;
	float x = (idx%by);
	x /= by;
	int b1 = center-1;
	int b2 = center-2;
	int a1 = center+1;
	int a2 = center+2;
	float y = lanczos(x, 2)*input[center];
	if(b1 >=0)
		y += lanczos(x+1, 2)*input[b1];
	if(b2 >=0)
		y += lanczos(x+2, 2)*input[b2];
	if(a1 < N)
		y += lanczos(x-1, 2)*input[a1];
	if(a2 < N)
		y += lanczos(x-2, 2)*input[a2];
	output[idx] = y;
}

#define SQ2P 2.506628274631000502415765284811

float gauss(float x, float d)
{
	float tmp = x/d;
	tmp *= tmp;
	return (1.0/(d*SQ2P))*exp(-0.5*tmp);
}

float gauss2(float x, float d)
{
	float tmp = (x-4.0/2)/(d*4.0/2);
	tmp *= tmp;
	return (1.0/(d*SQ2P))*exp(-0.5*tmp);
}

//#define MD 0.44721359549995793928183473374626
//#define MD 0.70710678118654752440084436210485
//#define MD 0.39937451095431716346804250415269
#define MD 0.4

__kernel void upSample2(int by, int N, __global float* input, __global float* output)
{
	int idx = get_global_id(0);
	int center = idx / by;
	float x = (idx%by);
	x /= by;
	int b1 = center-1;
	int b2 = center-2;
	int a1 = center+1;
	int a2 = center+2;
	float y = gauss2(x, MD)*input[center];
	if(b1 >=0)
		y += gauss2(x+1, MD)*input[b1];
	if(b2 >=0)
		y += gauss2(x+2, MD)*input[b2];
	if(a1 < N)
		y += gauss2(x-1, MD)*input[a1];
	if(a2 < N)
		y += gauss2(x-2, MD)*input[a2];
	output[idx] = y;
}