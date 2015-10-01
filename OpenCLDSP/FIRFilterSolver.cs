using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;
using System.Diagnostics;

namespace OpenCLDSP
{
    public unsafe class FIRFilterSolver : IDisposable
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initWorkBufferKernel { get; set; }
        private Kernel filterKernel { get; set; }
        private Kernel bufferedFilterKernel { get; set; }

        private Mem Table { get; set; }
        private Mem WorkBuffer { get; set; }
        private Mem OutputBuffer { get; set; }

        public int FilterOrder { get; private set; }
        public int CurrentPos { get; private set; }
        public int FilterCount { get; private set; }

        public FIRFilterSolver(Platform platform, IList<FIRFilter> filters) : this(platform, platform.QueryDevices(DeviceType.ALL)[0], filters)
        {

        }

        public FIRFilterSolver(Platform platform, Device device, IList<FIRFilter> filters)
        {
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
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(device);//, CommandQueueProperties.PROFILING_ENABLE);
            DiffEqnProgram = OpenCLContext.CreateProgramWithSource(File.OpenText("opencl/FIRKernels.cl").ReadToEnd());
            DiffEqnProgram.Build();
            initWorkBufferKernel = DiffEqnProgram.CreateKernel("initWorkBuffer");
            filterKernel = DiffEqnProgram.CreateKernel("filter");
            bufferedFilterKernel = DiffEqnProgram.CreateKernel("bufferedFilter");

            Table = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, table.Length * 4);
            WorkBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, table.Length * 4);
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, filters.Count * 4);
            
            fixed (float* array = table)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Table, false, 0, table.Length * 4, new IntPtr((void*)array));
            }
            initWorkBufferKernel.SetArg(0, WorkBuffer);
            OpenCLCommandQueue.EnqueueNDRangeKernel(initWorkBufferKernel, 1, null, new int[] { table.Length }, null);
            OpenCLCommandQueue.EnqueueBarrier();
            filterKernel.SetArg(2, FilterOrder);
            filterKernel.SetArg(3, Table);
            filterKernel.SetArg(4, WorkBuffer);
            filterKernel.SetArg(5, OutputBuffer);
            bufferedFilterKernel.SetArg(2, FilterOrder);
            bufferedFilterKernel.SetArg(3, Table);
            bufferedFilterKernel.SetArg(4, WorkBuffer);
            bufferedFilterKernel.SetArg(5, OutputBuffer);
            filterKernelGlobalWorkSize = new int[] { table.Length };
            //mapToOutputKernel.SetArg(1, FilterOrder);
            //mapToOutputKernel.SetArg(2, WorkBuffer);
            //mapToOutputKernel.SetArg(3, OutputBuffer);
            mapToOutputKernelGlobalWorkSize = new int[] { filters.Count };
            filterKernelLocalWorkSize = getLocalSize(filterKernelGlobalWorkSize[0]);
            mapToOutputLocalWorkSize = getLocalSize(mapToOutputKernelGlobalWorkSize[0]);
            SetOutputBufferLength(1);
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

        private Event LastStep;

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

        public void Perform(float input)
        {
            Event e1;
            filterKernel.SetArg(0, input);
            filterKernel.SetArg(1, CurrentPos);
            OpenCLCommandQueue.EnqueueNDRangeKernel(filterKernel, 1, null, filterKernelGlobalWorkSize, filterKernelLocalWorkSize, 1, new Event[] { LastStep }, out e1);
            //mapToOutputKernel.SetArg(0, CurrentPos);
            //OpenCLCommandQueue.EnqueueNDRangeKernel(mapToOutputKernel, 1, null, mapToOutputKernelGlobalWorkSize, mapToOutputLocalWorkSize, 1, new Event[] { e1 }, out LastStep);
            CurrentPos = (CurrentPos + 1) % FilterOrder;
            LastStep = e1;
            GC.Collect();
        }

        private int BufferLength { get; set; }
        private int BufferPos { get; set; }

        public void BufferedPerform(float input)
        {
            Event ev;
            bufferedFilterKernel.SetArg(0, input);
            bufferedFilterKernel.SetArg(1, CurrentPos);
            bufferedFilterKernel.SetArg(6, BufferPos);
            OpenCLCommandQueue.EnqueueNDRangeKernel(bufferedFilterKernel, 1, null, filterKernelGlobalWorkSize, filterKernelLocalWorkSize);//, 0, null, out ev);
            /*ulong start, end, queue;
            ev.Wait();
            var es = ev.ExecutionStatus;
            ev.GetEventProfilingInfo(ProfilingInfo.START, out start);
            ev.GetEventProfilingInfo(ProfilingInfo.QUEUED, out queue);
            ev.GetEventProfilingInfo(ProfilingInfo.END, out end);
            var endtime = end - start;
            var queuetime = start - queue;*/
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

        public float[] ReadCurrentOutput()
        {
            var output = new float[FilterCount];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length * 4, new IntPtr((void*)array), 1, new Event[]{LastStep});
            }
            return output;
        }

        public void Finish()
        {
            OpenCLCommandQueue.Finish();
        }
        /*
         private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initWorkBufferKernel { get; set; }
        private Kernel filterKernel { get; set; }
        private Kernel bufferedFilterKernel { get; set; }

        private Mem Table { get; set; }
        private Mem WorkBuffer { get; set; }
        private Mem OutputBuffer { get; set; }

        public int FilterOrder { get; private set; }
        public int CurrentPos { get; private set; }
        public int FilterCount { get; private set; }
         
         */
        public void Dispose()
        {
            //LastStep.Dispose();
            Table.Dispose();
            WorkBuffer.Dispose();
            OutputBuffer.Dispose();
            bufferedFilterKernel.Dispose();
            filterKernel.Dispose();
            initWorkBufferKernel.Dispose();
            DiffEqnProgram.Dispose();
            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
            
        }
    }
}
