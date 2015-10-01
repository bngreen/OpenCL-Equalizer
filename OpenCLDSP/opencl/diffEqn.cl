
typedef struct{
	int ReadPos;
	float PastInputs[DIFFEQNORDER];
	float PastOutputs[DIFFEQNORDER];
}DiffEqn;

__kernel void diffEqnSize(__global int* size)
{
	DiffEqn t;
	*size = sizeof(t);
}

__kernel void initDiffEqn(__global DiffEqn* diffEqn)
{
	int idx = get_global_id(0);
	diffEqn[idx].ReadPos = 0;
	for(int i=0;i<DIFFEQNORDER;i++)
	{
		diffEqn[idx].PastInputs[i] = 0;
		diffEqn[idx].PastOutputs[i] = 0;
	}
}

__kernel void performDiffEqn(__global DiffEqn* diffEqn, __global float* diffEqnInMuls, __global float* diffEqnOutMuls, float input, __global float* output)
{
	int idx = get_global_id(0);
	int diffEqnInd = idx / DIFFEQNORDER;
	int diffEqnPInd = idx % DIFFEQNORDER;
	if(diffEqnPInd == 0)
		output[idx] = input*diffEqnInMuls[idx];	
	else
	{
		int ind = (diffEqn[diffEqnInd].ReadPos - (diffEqnPInd - 1)) % DIFFEQNORDER;
		if(ind < 0)
			ind = DIFFEQNORDER + ind;
		float a = diffEqn[diffEqnInd].PastInputs[ind]*diffEqnInMuls[idx];
		float b = diffEqn[diffEqnInd].PastOutputs[ind]*diffEqnOutMuls[idx];
		output[idx] = a+b;
	}
}

__kernel void updateDiffEqn(__global DiffEqn* diffEqn, __global float* outputs, float input)
{
	int idx = get_global_id(0);
	diffEqn[idx].ReadPos = (diffEqn[idx].ReadPos + 1) % DIFFEQNORDER;
	diffEqn[idx].PastInputs[diffEqn[idx].ReadPos] = input;
	diffEqn[idx].PastOutputs[diffEqn[idx].ReadPos] = outputs[idx];
	
}

__kernel void performDiffEqn2(__global DiffEqn* diffEqn, __global float* diffEqnInMuls, __global float* diffEqnOutMuls, __global float* input, __global float* output)
{
	int idx = get_global_id(0);
	int diffEqnInd = idx / DIFFEQNORDER;
	int diffEqnPInd = idx % DIFFEQNORDER;
	if(diffEqnPInd == 0)
		output[idx] = input[diffEqnInd]*diffEqnInMuls[idx];	
	else
	{
		int ind = (diffEqn[diffEqnInd].ReadPos - (diffEqnPInd - 1)) % DIFFEQNORDER;
		if(ind < 0)
			ind = DIFFEQNORDER + ind;
		float a = diffEqn[diffEqnInd].PastInputs[ind]*diffEqnInMuls[idx];
		float b = diffEqn[diffEqnInd].PastOutputs[ind]*diffEqnOutMuls[idx];
		output[idx] = a+b;
	}
}

__kernel void updateDiffEqn2(__global DiffEqn* diffEqn, __global float* outputs, __global float* input)
{
	int idx = get_global_id(0);
	diffEqn[idx].ReadPos = (diffEqn[idx].ReadPos + 1) % DIFFEQNORDER;
	diffEqn[idx].PastInputs[diffEqn[idx].ReadPos] = input[idx];
	diffEqn[idx].PastOutputs[diffEqn[idx].ReadPos] = outputs[idx];
}

__kernel void sumReduceN(__global float* input, __global float* output, int N)
{
	int idx = get_global_id(0);
	output[idx] = 0;
	for(int i=0;i<N;i++)
		output[idx]+=input[idx*N+i];
}

__kernel void sumReduce(__global float* input, __global float* output)
{
	int idx = get_global_id(0);
	int inInd = idx*2;
	output[idx] = input[inInd] + input[inInd+1];
}

__kernel void constMultiply(__global float* input, __global float* v, __global float* output)
{
	int idx = get_global_id(0);
	output[idx] = input[idx]*v[idx];
}



__kernel void test(float v, __global float* o, __global float* t)
{
	float y = 0.0639f * v - 0.1251f * t[1] + 0.0639f * t[2] - 1.1741f * t[3] - 0.4269f * t[4];
	t[4] = t[3];
    t[3] = y;
    t[2] = t[1];
    t[1] = v;
	o[0] = y;
}