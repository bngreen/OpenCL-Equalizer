__kernel void initWorkBuffer(__global float* workBuffer)
{
	int idx = get_global_id(0);
	workBuffer[idx] = 0;
}


__kernel void bufferedFilter(float input, int currentPos, int filterOrder, __global float* table, __global float* workBuffer, __global float* output, int outputPos, int outputLen)
{
	int idx = get_global_id(0);
	int flt = idx / filterOrder;
	int pos = idx % filterOrder;
	int index = (pos + currentPos) % filterOrder;
	float value = table[flt*filterOrder+pos]*input;
	index = flt*filterOrder + index;
	float oldval = workBuffer[index];
	if(pos != (filterOrder-1))
		value += oldval;
	workBuffer[index] = value;
	if(pos == 0)
		output[flt*outputLen+outputPos] = workBuffer[index];
}

__kernel void stage2(__global float* workBuffer, __global float* table, int currentPos, int filterOrder)
{
	int idx = get_global_id(0);
	int flt = idx / (filterOrder-1);
	int pos = idx % (filterOrder-1);
	pos +=1;
	int index = (pos + currentPos) % filterOrder;
	float y = workBuffer[flt*filterOrder + currentPos];
	int ind = flt*filterOrder+index;
	workBuffer[ind] -= y*table[flt*filterOrder+pos];
} 