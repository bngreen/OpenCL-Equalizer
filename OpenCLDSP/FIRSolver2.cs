using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;

namespace OpenCLDSP
{
    public unsafe class FIRSolver2
    {
    private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initInputBufferKernel { get; set; }
        private Kernel filter1 { get; set; }
        private Kernel filter2 { get; set; }
        private Kernel mulAndSumKernel { get; set; }

        private Mem Table { get; set; }
        private Mem Table2 { get; set; }
        private Mem OutputBuffer { get; set; }
        private Mem Input2 { get; set; }
        private Mem Input1 { get; set; }
        private Mem OutputBuffer2 { get; set; }


        public int FilterOrder { get; private set; }
        public int FilterCount { get; private set; }

        public FIRSolver2(Platform platform, IList<FIRFilter> filters, int N) : this(platform, platform.QueryDevices(DeviceType.ALL)[0], filters, N)
        {

        }

        public FIRSolver2(Platform platform, Device device, IList<FIRFilter> filters, int N)
        {
            BufferLength = N;
            FilterCount = filters.Count;
            var order = filters[0].B.Count;
            FilterOrder = order;
            foreach (var x in filters)
                if (order != x.B.Count)
                    throw new InvalidOperationException("The filters should have the same order");
            var table = new float[filters.Count * order];
            var f = 0;
            foreach (var x in filters)
            {
                for (int i = 0; i < order; i++)
                    table[f*order + i] = x.B[i];
                f++;
            }
            Platform = platform;
            OpenCLContext = Platform.CreateDefaultContext();
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(device);
            DiffEqnProgram = OpenCLContext.CreateProgramWithSource(File.OpenText("opencl/FIRKernels2.cl").ReadToEnd());
            DiffEqnProgram.Build();
            initInputBufferKernel = DiffEqnProgram.CreateKernel("initInputBuffer");
            filter1 = DiffEqnProgram.CreateKernel("bufferedFilter");
            filter2 = DiffEqnProgram.CreateKernel("bufferedFilter2");
            mulAndSumKernel = DiffEqnProgram.CreateKernel("mulAndSum");

            Table = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, table.Length * 4);
            Table2 = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, FilterCount * 4);
            Input1 = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, N * 4);
            Input2 = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, N * 4);
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, filters.Count * 4 * N);
            OutputBuffer2 = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, N * 4);
            fixed (float* array = table)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Table, false, 0, table.Length * 4, new IntPtr((void*)array));
            }
            initInputBufferKernel.SetArg(0, Input2);
            OpenCLCommandQueue.EnqueueNDRangeKernel(initInputBufferKernel, 1, null, new int[] { N }, null);
            OpenCLCommandQueue.EnqueueBarrier();

            filter1.SetArg(2, FilterOrder);
            filter1.SetArg(3, Table);
            filter1.SetArg(4, OutputBuffer);
            filter1.SetArg(5, N);
            filter2.SetArg(2, FilterOrder);
            filter2.SetArg(3, Table);
            filter2.SetArg(4, OutputBuffer);
            filter2.SetArg(5, N);

            filterKernelGlobalWorkSize = new int[] { N*FilterCount };
            filterLocalWorkSize = getLocalSize(filterKernelGlobalWorkSize[0]);

            mulAndSumKernel.SetArg(0, OutputBuffer);
            mulAndSumKernel.SetArg(1, Table2);
            mulAndSumKernel.SetArg(2, OutputBuffer2);
            mulAndSumKernel.SetArg(3, FilterCount);
            mulAndSumKernel.SetArg(4, BufferLength);
        }

        public void SetMulTable(float[] table)
        {
            fixed (float* array = table)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Table2, false, 0, table.Length*4, new IntPtr((void*)array));
                OpenCLCommandQueue.EnqueueBarrier();
            }
        }

        public float[] PerformAndSum(float[] input)
        {
            if (input.Length != BufferLength)
                throw new InvalidOperationException();
            float[] output = new float[BufferLength];
            fixed (float* array1 = input, array2 = output)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Input1, false, 0, BufferLength * 4, new IntPtr((void*)array1));
                OpenCLCommandQueue.EnqueueBarrier();
                filter1.SetArg(0, Input2);
                filter1.SetArg(1, Input1);
                OpenCLCommandQueue.EnqueueNDRangeKernel(filter1, 1, null, filterKernelGlobalWorkSize, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueNDRangeKernel(mulAndSumKernel, 1, null, new int[] { BufferLength }, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer2, true, 0, output.Length * 4, new IntPtr((void*)array2));

                Mem temp = Input1;
                Input1 = Input2;
                Input2 = temp;
            }
            return output;
        }

        public float[] Perform(float[] input)
        {
            if (input.Length != BufferLength)
                throw new InvalidOperationException();
            float[] output = new float[BufferLength * FilterCount];
            fixed (float* array1 = input, array2 = output)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Input1, false, 0, BufferLength * 4, new IntPtr((void*)array1));
                OpenCLCommandQueue.EnqueueBarrier();
                filter1.SetArg(0, Input2);
                filter1.SetArg(1, Input1);
                OpenCLCommandQueue.EnqueueNDRangeKernel(filter1, 1, null, filterKernelGlobalWorkSize, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length * 4, new IntPtr((void*)array2));

                Mem temp = Input1;
                Input1 = Input2;
                Input2 = temp;
            }
            return output;
        }

        public float[] Perform2(float[] input)
        {
            if (input.Length != BufferLength)
                throw new InvalidOperationException();
            float[] output = new float[BufferLength * FilterCount];
            fixed (float* array1 = input, array2 = output)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Input1, false, 0, BufferLength * 4, new IntPtr((void*)array1));
                OpenCLCommandQueue.EnqueueBarrier();
                filter2.SetArg(0, Input2);
                filter2.SetArg(1, Input1);
                OpenCLCommandQueue.EnqueueNDRangeKernel(filter2, 1, null, filterKernelGlobalWorkSize, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length * 4, new IntPtr((void*)array2));

                Mem temp = Input1;
                Input1 = Input2;
                Input2 = temp;
            }
            return output;
        }

        private int[] filterKernelGlobalWorkSize { get; set; }

        private int[] filterLocalWorkSize { get; set; }

        private int[] getLocalSize(int sz)
        {
            //return null;
            var val = 256;
            for (; val > 0; val--)
            {
                if (sz % val == 0)
                    break;
            }
            return new int[] { val };
        }

        private int BufferLength { get; set; }

        public float[] ReadOutputBuffer()
        {
            var output = new float[FilterCount * BufferLength];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length * 4, new IntPtr((void*)array));
            }
            return output;
        }

        public void Finish()
        {
            OpenCLCommandQueue.Finish();
        }

        public void Dispose()
        {
            Table.Dispose();
            Input1.Dispose();
            Input2.Dispose();
            OutputBuffer.Dispose();
            filter1.Dispose();
            filter2.Dispose();
            initInputBufferKernel.Dispose();
            DiffEqnProgram.Dispose();
            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
            
        }
    }
}
