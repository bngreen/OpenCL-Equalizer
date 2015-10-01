using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;

namespace OpenCLDSP
{
    public unsafe class BufferedFIRSolver
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        public IList<Device> OpenCLDevices { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initWorkBufferKernel { get; set; }
        private Kernel filterKernel { get; set; }
        private Kernel bufferedFilterKernel { get; set; }

        private Mem Table { get; set; }
        private Mem WorkBuffer { get; set; }
        private Mem OutputBuffer { get; set; }
        private Mem InputBuffer { get; set; }
        private Mem LockBuffer { get; set; }

        public int FilterOrder { get; private set; }
        public int CurrentPos { get; private set; }
        public int FilterCount { get; private set; }

        private int BufferLength { get; set; }
        private int BufferPos { get; set; }

        public BufferedFIRSolver(Platform platform, IList<FIRFilter> filters, int bufferLength)
        {
            BufferLength = bufferLength;
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
            OpenCLDevices = Platform.QueryDevices(DeviceType.ALL);
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(OpenCLDevices[0]);
            DiffEqnProgram = OpenCLContext.CreateProgramWithSource(File.OpenText("opencl/BufferedFIRFilter.cl").ReadToEnd());
            DiffEqnProgram.Build();
            initWorkBufferKernel = DiffEqnProgram.CreateKernel("initWorkBuffer");
            filterKernel = DiffEqnProgram.CreateKernel("filter");

            Table = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, table.Length * 4);
            WorkBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, table.Length * 4);
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, bufferLength*filters.Count * 4);
            InputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, bufferLength * filters.Count * 4);
            LockBuffer = OpenCLContext.CreateBuffer(MemFlags.READ_ONLY, FilterCount * 4);

            fixed (float* array = table)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(Table, false, 0, table.Length * 4, new IntPtr((void*)array));
            }
            initWorkBufferKernel.SetArg(0, WorkBuffer);
            OpenCLCommandQueue.EnqueueNDRangeKernel(initWorkBufferKernel, 1, null, new int[] { table.Length }, null, 0, null, out LastStep);

            filterKernel.SetArg(0, InputBuffer);
            filterKernel.SetArg(2, FilterOrder);
            filterKernel.SetArg(3, Table);
            filterKernel.SetArg(4, WorkBuffer);
            filterKernel.SetArg(5, OutputBuffer);
            filterKernel.SetArg(6, BufferLength);
            filterKernel.SetArg(7, LockBuffer);

            filterKernelGlobalWorkSize = new int[] { table.Length };
            mapToOutputKernelGlobalWorkSize = new int[] { filters.Count };
            filterKernelLocalWorkSize = getLocalSize(filterKernelGlobalWorkSize[0]);
            mapToOutputLocalWorkSize = getLocalSize(mapToOutputKernelGlobalWorkSize[0]);

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

        public void Perform(float[] input)
        {
            if (input.Length != BufferLength)
                throw new InvalidOperationException("Invalid Input Length");
            Event writeBufferevn;
            fixed (float* array = input)
            {

                OpenCLCommandQueue.EnqueueWriteBuffer(InputBuffer, false, 0, input.Length * 4, new IntPtr((void*)array), 1, new Event[] { LastStep }, out writeBufferevn);
            }
            Event e1;
            filterKernel.SetArg(1, CurrentPos);
            OpenCLCommandQueue.EnqueueNDRangeKernel(filterKernel, 1, null, filterKernelGlobalWorkSize, filterKernelLocalWorkSize, 1, new Event[] { writeBufferevn }, out e1);
            //mapToOutputKernel.SetArg(0, CurrentPos);
            //OpenCLCommandQueue.EnqueueNDRangeKernel(mapToOutputKernel, 1, null, mapToOutputKernelGlobalWorkSize, mapToOutputLocalWorkSize, 1, new Event[] { e1 }, out LastStep);
            CurrentPos = (CurrentPos + BufferLength) % FilterOrder;
            LastStep = e1;
        }

        public float[] ReadOutputBuffer()
        {
            var output = new float[FilterCount * BufferLength];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length * 4, new IntPtr((void*)array), 1, new Event[] { LastStep });
            }
            return output;
        }

        public void Finish()
        {
            OpenCLCommandQueue.Finish();
        }
    }
}
