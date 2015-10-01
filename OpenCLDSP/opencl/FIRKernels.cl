
__kernel void initWorkBuffer(__global float* workBuffer)
{
	int idx = get_global_id(0);
	workBuffer[idx] = 0;
}


__kernel void filter(float input, int currentPos, int filterOrder, __global float* table, __global float* workBuffer, __global float* output)
{
	int idx = get_global_id(0);
	int flt = idx / filterOrder;
	int pos = idx % filterOrder;
	int index = (pos + currentPos) % filterOrder;
	float value = table[flt*filterOrder+pos]*input;
	if(pos == (filterOrder-1))
		workBuffer[flt*filterOrder + index] = value;
	else
		workBuffer[flt*filterOrder + index] += value;
	if(pos == 0)
		output[flt] = workBuffer[flt*filterOrder + index];
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

__kernel void bufferedFilter2(__global float* inputBuff,__global int* posBuff, int filterOrder, __global float* table, __global float* workBuffer, __global float* output, int outputLen)
{
	int idx = get_global_id(0);
	int flt = idx / filterOrder;
	int pos = idx % filterOrder;
	int currentPos = posBuff[0];
	int outputPos = posBuff[1];
	int index = (pos + currentPos) % filterOrder;
	float input = inputBuff[outputPos];
	float value = table[pos]*input;
	index = flt*filterOrder + index;
	float oldval = workBuffer[index];
	if(pos != (filterOrder-1))
		value += oldval;
	workBuffer[index] = value;
	if(pos == 0)
		output[flt*outputLen+outputPos] = workBuffer[index];
}

__kernel void incrementPosBuff(__global int* posBuff, int filterOrder, int outputLen)
{
	posBuff[0] = (posBuff[0] + 1)%filterOrder;
	posBuff[1] = (posBuff[1] + 1)%outputLen;
}

__kernel void mapToOutput(int currentPos, int filterOrder, __global float* workBuffer, __global float* output)
{
	int idx = get_global_id(0);
	int dataIndex = (currentPos % filterOrder) + idx*filterOrder;
	output[idx] = workBuffer[dataIndex];
}

