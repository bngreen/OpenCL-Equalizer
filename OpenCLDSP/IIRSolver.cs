using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;

namespace OpenCLDSP
{
    public unsafe class IIRSolver : IDisposable
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initWorkBufferKernel { get; set; }
        private Kernel bufferedFilterKernel { get; set; }
        private Kernel stage2Kernel { get; set; }

        private Mem Table { get; set; }
        private Mem Table2 { get; set; }
        private Mem WorkBuffer { get; set; }
        private Mem OutputBuffer { get; set; }

        public int FilterOrder { get; private set; }
        public int CurrentPos { get; private set; }
        public int FilterCount { get; private set; }

        public IIRSolver(Platform platform, IList<IIRFilter> filters) : this(platform, platform.QueryDevices(DeviceType.ALL)[0], filters)
        {

        }

        public IIRSolver(Platform platform, Device device, IList<IIRFilter> filters)
        {
            FilterCount = filters.Count;
            var order = filters[0].B.Count;
            FilterOrder = order;
            foreach (var x in filters)
                if (order != x.B.Count || order != x.A.Count)
                    throw new InvalidOperationException("The filters should have the same order");
            var table = new float[filters.Count * order];
            var table2 = new float[filters.Count * order];
            var f = 0;
            foreach (var x in filters)
            {
                for (int i = 0; i < order; i++)
                {
                    table[f * order + i] = x.B[i];
                    table2[f * order + i] = x.A[i];
                }
                f++;
            }
            Platform = platform;
            OpenCLContext = Platform.CreateDefaultContext();
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(device);//, CommandQueueProperties.PROFILING_ENABLE);
            DiffEqnProgram = OpenCLContext.CreateProgramWithSource(File.OpenText("opencl/IIRKernel.cl").ReadToEnd());
            DiffEqnProgram.Build();
            initWorkBufferKernel = DiffEqnProgram.CreateKernel("initWorkBuffer");
            stage2Kernel = DiffEqnProgram.CreateKernel("stage2");
            bufferedFilterKernel = DiffEqnProgram.CreateKernel("bufferedFilter");

            Table = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, table.Length * 4);
            Table2 = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, table.Length * 4);
            WorkBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, table.Length * 4);
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, filters.Count * 4);
            
            fixed (float* array = table, array2 = table2)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Table, false, 0, table.Length * 4, new IntPtr((void*)array));
                OpenCLCommandQueue.EnqueueWriteBuffer(Table2, false, 0, table2.Length * 4, new IntPtr((void*)array2));
            }
            initWorkBufferKernel.SetArg(0, WorkBuffer);
            OpenCLCommandQueue.EnqueueNDRangeKernel(initWorkBufferKernel, 1, null, new int[] { table.Length }, null);
            OpenCLCommandQueue.EnqueueBarrier();

            bufferedFilterKernel.SetArg(2, FilterOrder);
            bufferedFilterKernel.SetArg(3, Table);
            bufferedFilterKernel.SetArg(4, WorkBuffer);
            bufferedFilterKernel.SetArg(5, OutputBuffer);

            stage2Kernel.SetArg(0, WorkBuffer);
            stage2Kernel.SetArg(1, Table2);
            
            stage2Kernel.SetArg(3, FilterOrder);

            filterKernelGlobalWorkSize = new int[] { table.Length };
            mapToOutputKernelGlobalWorkSize = new int[] { filters.Count };
            filterKernelLocalWorkSize = getLocalSize(filterKernelGlobalWorkSize[0]);
            mapToOutputLocalWorkSize = getLocalSize(mapToOutputKernelGlobalWorkSize[0]);
            SetOutputBufferLength(1);
            stage2GlobalWorkSize = new int[] { FilterCount * (FilterOrder - 1) };
        }

        public void SetOutputBufferLength(int len)
        {
            OpenCLCommandQueue.Finish();
            OutputBuffer.Dispose();
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, 4 * FilterCount * len);
            BufferLength = len;
            bufferedFilterKernel.SetArg(7, len);
            bufferedFilterKernel.SetArg(5, OutputBuffer);
        }

        private int[] stage2GlobalWorkSize { get; set; }

        private int[] filterKernelGlobalWorkSize { get; set; }
        private int[] mapToOutputKernelGlobalWorkSize { get; set; }

        private int[] filterKernelLocalWorkSize { get; set; }
        private int[] mapToOutputLocalWorkSize { get; set; }

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
        private int BufferPos { get; set; }

        public void BufferedPerform(float input)
        {
            bufferedFilterKernel.SetArg(0, input);
            bufferedFilterKernel.SetArg(1, CurrentPos);
            bufferedFilterKernel.SetArg(6, BufferPos);
            stage2Kernel.SetArg(2, CurrentPos);
            OpenCLCommandQueue.EnqueueNDRangeKernel(bufferedFilterKernel, 1, null, filterKernelGlobalWorkSize, filterKernelLocalWorkSize);//, 0, null, out ev);
            OpenCLCommandQueue.EnqueueBarrier();
            OpenCLCommandQueue.EnqueueNDRangeKernel(stage2Kernel, 1, null, stage2GlobalWorkSize, null);
            OpenCLCommandQueue.EnqueueBarrier();
            CurrentPos = (CurrentPos + 1) % FilterOrder;
            BufferPos = (BufferPos + 1) % BufferLength;
        }

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
            Table2.Dispose();
            Table.Dispose();
            WorkBuffer.Dispose();
            OutputBuffer.Dispose();
            bufferedFilterKernel.Dispose();
            stage2Kernel.Dispose();
            initWorkBufferKernel.Dispose();
            DiffEqnProgram.Dispose();
            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
            
        }
    }
}
