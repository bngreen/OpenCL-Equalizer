
__kernel void initWorkBuffer(__global float* workBuffer)
{
	int idx = get_global_id(0);
	workBuffer[idx] = 0;
}


void waitOthers(volatile __global int* lock, int itemCount, bool even)
{
	if(even)
	{
		atomic_inc(lock);
		while(*lock < itemCount);
	}
	else
	{
		atomic_dec(lock);
		while(*lock > 0);
	}
}

__kernel void filter(__global float* input, int currentPos, int filterOrder, __global float* table, __global float* workBuffer, __global float* output, int N, volatile __global int* lock)
{
	int idx = get_global_id(0);
	int flt = idx / filterOrder;
	int pos = idx % filterOrder;
	lock[flt] = 0;
	bool even = true;
	for(int i=0;i<N;i++)
	{

		int index = (pos + currentPos + i) % filterOrder;
		float value = table[pos]*input[flt*N + i];
		if(pos == (filterOrder-1))
			workBuffer[flt*filterOrder + index] = value;
		else
			workBuffer[flt*filterOrder + index] += value;
		if(pos == 0)
			output[flt*N+i] = workBuffer[flt*filterOrder + index];
		
		waitOthers(&lock[flt], filterOrder, even);

		even = !even;

	}
}