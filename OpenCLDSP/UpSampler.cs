using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;

namespace OpenCLDSP
{
    public unsafe class UpSampler : IDisposable
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        private Program UpSampleProgram { get; set; }
        private Kernel upsampleKernel { get; set; }
        private Kernel upsample2Kernel { get; set; }

        private Mem InputBuffer { get; set; }
        private Mem OutputBuffer { get; set; }

        public int InputBufferLen { get; set; }
        public int By { get; set; }

        public UpSampler(Platform platform, int by, int inputBufferLen)
            : this(platform, platform.QueryDevices(DeviceType.ALL)[0], by, inputBufferLen)
        {

        }

        public void Dispose()
        {
            InputBuffer.Dispose();
            OutputBuffer.Dispose();
            upsampleKernel.Dispose();
            upsample2Kernel.Dispose();
            UpSampleProgram.Dispose();
            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
        }

        public UpSampler(Platform platform, Device device, int by, int inputBufferLen)
        {
            InputBufferLen = inputBufferLen;
            By = by;
            Platform = platform;
            OpenCLContext = Platform.CreateDefaultContext();
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(device);
            UpSampleProgram = OpenCLContext.CreateProgramWithSource(File.OpenText("opencl/upSample.cl").ReadToEnd());
            UpSampleProgram.Build();
            upsampleKernel = UpSampleProgram.CreateKernel("upSample");
            upsample2Kernel = UpSampleProgram.CreateKernel("upSample2");
            upsampleKernel.SetArg(0, by);
            upsampleKernel.SetArg(1, inputBufferLen);
            upsample2Kernel.SetArg(0, by);
            upsample2Kernel.SetArg(1, inputBufferLen);


            InputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, inputBufferLen * 4);
            OutputBuffer = OpenCLContext.CreateBuffer(MemFlags.WRITE_ONLY, inputBufferLen * by * 4);
            upsampleKernel.SetArg(2, InputBuffer);
            upsampleKernel.SetArg(3, OutputBuffer);
            upsample2Kernel.SetArg(2, InputBuffer);
            upsample2Kernel.SetArg(3, OutputBuffer);
            globalworksize = new int[] { (int)(by * inputBufferLen) };
        }

        private int[] globalworksize { get; set; }

        public float[] Perform(float[] input)
        {
            if (input.Length != InputBufferLen)
                throw new InvalidOperationException();
            var output = new float[InputBufferLen * By];
            fixed (float* array1 = input, array2 = output)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(InputBuffer, true, 0, InputBufferLen * 4, new IntPtr((void*)array1));
                //OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueNDRangeKernel(upsampleKernel, 1, null, globalworksize, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length, new IntPtr((void*)array2));
            }
            return output;
        }

        public float[] Perform2(float[] input)
        {
            if (input.Length != InputBufferLen)
                throw new InvalidOperationException();
            var output = new float[InputBufferLen * By];
            fixed (float* array1 = input, array2 = output)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(InputBuffer, true, 0, InputBufferLen * 4, new IntPtr((void*)array1));
                //OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueNDRangeKernel(upsample2Kernel, 1, null, globalworksize, null);
                OpenCLCommandQueue.EnqueueBarrier();
                OpenCLCommandQueue.EnqueueReadBuffer(OutputBuffer, true, 0, output.Length, new IntPtr((void*)array2));
            }
            return output;
        }

    }
}
