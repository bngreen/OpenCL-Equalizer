
__kernel void initInputBuffer(__global float* buff)
{
	int idx = get_global_id(0);
	buff[idx] = 0;
}

__kernel void bufferedFilter(__global float* lastInput, __global float* input, int filterOrder, __global float* table, __global float* output, int N)
{
	int idx = get_global_id(0);
	int flt = idx / N;
	int ind = idx % N;
	float val = 0;
	for(int i=0;i<filterOrder;i++)
	{
		int index = ind - i;
		float b = table[flt*filterOrder + i];
		if(index < 0)
		{
			index += N;
			val += b*lastInput[index];
		}
		else
			val += b*input[index];
	}
	output[idx] = val;
}

__kernel void bufferedFilter2(__global float* lastInput, __global float* input, int filterOrder, __global float* table, __global float* output, int N)
{
	int idx = get_global_id(0);
	int flt = idx / N;
	int ind = idx % N;
	float val = 0;
	for(int i=0;i<filterOrder;i++)
	{
		int index = ind - i;
		float b = table[flt*filterOrder + i];
		if(index < 0)
		{
			index += N;
			val += b*lastInput[index];
		}
		else
			val += b*input[index];
		index = ind + i;
		if(index < N && i > 0)
			val += b*input[index];
	}
	output[idx] = val;
}

__kernel void mulAndSum(__global float* input, __global float* table, __global float* output, int filters, int N)
{
	//int N = get_global_size(0);
	int idx = get_global_id(0);
	output[idx] = 0;
	//output[idx] = input[idx]*table[0];
	for(int i=0;i<filters;i++)
		output[idx] += input[i*N+idx] * table[i];
}