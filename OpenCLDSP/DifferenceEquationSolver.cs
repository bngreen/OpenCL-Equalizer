using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;
using System.Diagnostics;
namespace OpenCLDSP
{
    public unsafe class DifferenceEquationSolver : IDisposable
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        public IList<Device> OpenCLDevices { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initDiffEqn { get; set; }
        private Kernel performDiffEqn { get; set; }
        private Kernel updateDiffEqn { get; set; }
        private Kernel sumReduce { get; set; }
        private Kernel constMultiply { get; set; }
        private Kernel testKernel { get; set; }
        private Kernel diffEqnSizeKernel { get; set; }
        private Kernel sumReduceN { get; set; }

        private Mem DiffEquationsInstances { get; set; }

        private IList<Mem> CollapsingInstances { get; set; }
        private Mem InputMulInstance { get; set; }
        private Mem OutputMulInstance { get; set; }
        public int DiffEqnOrder { get; private set; }
        public int DiffEqnCount { get; private set; }
        public int DiffEqnSize { get; set; }

        public unsafe DifferenceEquationSolver(Platform platform, int diffEqnOrder, IList<DifferenceEquation> diffEquations)
        {
            DiffEqnCount = diffEquations.Count;
            DiffEqnOrder = diffEqnOrder;
            var exponent = (int)Math.Round(Math.Log(diffEqnOrder, 2));
            if (Math.Abs(diffEqnOrder - Math.Pow(2, exponent)) > 0.001)
                throw new InvalidOperationException("diffEqnOrder must be power of base 2");
            var inMul = new float[diffEquations.Count * diffEqnOrder];
            var outMul = new float[diffEquations.Count * diffEqnOrder];
            var dInd = 0;
            foreach (var x in diffEquations)
            {
                if (x.B.Count != diffEqnOrder || x.A.Count != diffEqnOrder)
                    throw new InvalidOperationException();
                if (Math.Abs(1-x.A[0]) > 0.0000001)
                {
                    var b = x.A[0];
                    for (int i = 0; i < x.B.Count; i++)
                    {
                        x.B[i] /= b;
                        x.A[i] /= b;
                    }
                    x.A[0] = 1;
                }
                for (int i = 0; i < diffEqnOrder; i++)
                {
                    var index = dInd * diffEqnOrder + i;
                    inMul[index] = x.B[i];
                    outMul[index] = x.A[i];
                }
                dInd++;
            }


            Platform = platform;
            OpenCLContext = Platform.CreateDefaultContext();
            OpenCLDevices = Platform.QueryDevices(DeviceType.ALL);
            OpenCLCommandQueue = OpenCLContext.CreateCommandQueue(OpenCLDevices[0]);
            DiffEqnProgram = OpenCLContext.CreateProgramWithSource("#define DIFFEQNORDER " + diffEqnOrder + "\n" + File.OpenText("opencl/diffEqn.cl").ReadToEnd());
            DiffEqnProgram.Build();
            initDiffEqn = DiffEqnProgram.CreateKernel("initDiffEqn");
            performDiffEqn = DiffEqnProgram.CreateKernel("performDiffEqn");
            updateDiffEqn = DiffEqnProgram.CreateKernel("updateDiffEqn");
            sumReduce = DiffEqnProgram.CreateKernel("sumReduce");
            constMultiply = DiffEqnProgram.CreateKernel("constMultiply");
            testKernel = DiffEqnProgram.CreateKernel("test");
            diffEqnSizeKernel = DiffEqnProgram.CreateKernel("diffEqnSize");
            sumReduceN = DiffEqnProgram.CreateKernel("sumReduceN");


            CollapsingInstances = new List<Mem>();
            for (int i = exponent; i >= 0; i--)
                CollapsingInstances.Add(OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4*(int)Math.Pow(2, i)*diffEquations.Count));

            var val = new int[1];

            fixed (int* num = val)
            {
                diffEqnSizeKernel.SetArg(0, CollapsingInstances[0]);
                OpenCLCommandQueue.EnqueueNDRangeKernel(diffEqnSizeKernel, 1, null, new int[1] { 1 }, null);
                OpenCLCommandQueue.EnqueueReadBuffer(CollapsingInstances[0], true, 0, 4, new IntPtr((void*)num));
                OpenCLCommandQueue.Finish();
            }
            DiffEqnSize = val[0];

            DiffEquationsInstances = OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, (4 + 8 * diffEqnOrder) * DiffEqnCount);

            initDiffEqn.SetArg(0, DiffEquationsInstances);
            OpenCLCommandQueue.EnqueueNDRangeKernel(initDiffEqn, 1, null, new int[1] { DiffEqnCount }, null);
            OpenCLCommandQueue.Flush();
            OpenCLCommandQueue.Finish();

            


            InputMulInstance = OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * inMul.Length);
            OutputMulInstance = OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * outMul.Length);

            fixed (float* array1 = inMul, array2 = outMul)
            {
                OpenCLCommandQueue.EnqueueWriteBuffer(InputMulInstance, true, 0, 4 * inMul.Length, new IntPtr((void*)array1));
                OpenCLCommandQueue.EnqueueWriteBuffer(OutputMulInstance, true, 0, 4 * outMul.Length, new IntPtr((void*)array2));
                OpenCLCommandQueue.Finish();
            }

            performDiffEqn.SetArg(0, DiffEquationsInstances);
            performDiffEqn.SetArg(1, InputMulInstance);
            performDiffEqn.SetArg(2, OutputMulInstance);
            performDiffEqn.SetArg(4, CollapsingInstances[0]);

            updateDiffEqn.SetArg(0, DiffEquationsInstances);
            updateDiffEqn.SetArg(1, CollapsingInstances.Last());
            sizes = new List<int[]>();
            localsizes = new List<int[]>();
            for (int i = 0; i < CollapsingInstances.Count - 1; i++)
            {
                sizes.Add(new int[] { (int)Math.Pow(2, CollapsingInstances.Count - i - 2) * DiffEqnCount });
                localsizes.Add(getLocalSize(sizes[i][0]));
            }

            sumReduceN.SetArg(0, CollapsingInstances[0]);
            sumReduceN.SetArg(1, CollapsingInstances.Last());
            sumReduceN.SetArg(2, DiffEqnOrder);


        }

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

        private IList<int[]> sizes;
        private IList<int[]> localsizes;

        public unsafe void Perform2(float input)
        {
            

            performDiffEqn.SetArg(3, input);
            
            OpenCLCommandQueue.EnqueueNDRangeKernel(performDiffEqn, 1, null, new int[] { DiffEqnCount * DiffEqnOrder }, getLocalSize(DiffEqnCount * DiffEqnOrder));
            OpenCLCommandQueue.Finish();
            OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduceN, 1, null, new int[] { DiffEqnCount }, null);
            OpenCLCommandQueue.Finish();
            /*for (int i = 0; i < CollapsingInstances.Count - 1; i++)
            {
                sumReduce.SetArg(0, CollapsingInstances[i]);
                sumReduce.SetArg(1, CollapsingInstances[i + 1]);
                
                OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduce, 1, null, sizes[i] , localsizes[i]);
                OpenCLCommandQueue.Finish();
            }*/
            updateDiffEqn.SetArg(2, input);
            OpenCLCommandQueue.EnqueueNDRangeKernel(updateDiffEqn, 1, null, new int[] { DiffEqnCount }, getLocalSize(DiffEqnCount));
            OpenCLCommandQueue.Finish();

        }

        public unsafe void Perform(float input)
        {


            performDiffEqn.SetArg(3, input);

            OpenCLCommandQueue.EnqueueNDRangeKernel(performDiffEqn, 1, null, new int[] { DiffEqnCount * DiffEqnOrder }, getLocalSize(DiffEqnCount * DiffEqnOrder));
            OpenCLCommandQueue.EnqueueBarrier();
            OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduceN, 1, null, new int[] { DiffEqnCount }, null);
            OpenCLCommandQueue.EnqueueBarrier();
            /*for (int i = 0; i < CollapsingInstances.Count - 1; i++)
            {
                sumReduce.SetArg(0, CollapsingInstances[i]);
                sumReduce.SetArg(1, CollapsingInstances[i + 1]);
                
                OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduce, 1, null, sizes[i] , localsizes[i]);
                OpenCLCommandQueue.Finish();
            }*/
            updateDiffEqn.SetArg(2, input);
            OpenCLCommandQueue.EnqueueNDRangeKernel(updateDiffEqn, 1, null, new int[] { DiffEqnCount }, getLocalSize(DiffEqnCount));
            OpenCLCommandQueue.Finish();

        }

        public unsafe Tuple<float, float[]> Test2(float v)
        {
            var output = new float[1];
            var test = new float[9];
            fixed (float* array = output, array2 = test)
            {
                testKernel.SetArg(0, v);
                testKernel.SetArg(1, CollapsingInstances.First());
                testKernel.SetArg(2, DiffEquationsInstances);
                //OpenCLCommandQueue.EnqueueNDRangeKernel(testKernel, 1, null, new int[] { 1 }, null);
                OpenCLCommandQueue.EnqueueReadBuffer(CollapsingInstances.First(), true, 0, 1 * 4, new IntPtr((void*)array));
                OpenCLCommandQueue.EnqueueReadBuffer(DiffEquationsInstances, true, 0, test.Length * 4, new IntPtr((void*)array2));
                OpenCLCommandQueue.Finish();
            }
            return new Tuple<float,float[]>(output[0], test);
        }

        public unsafe float[] Test()
        {
            var output = new float[DiffEqnOrder * DiffEqnCount];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(CollapsingInstances.First(), true, 0, output.Length * 4, new IntPtr((void*)array));
                OpenCLCommandQueue.Finish();
            }
            return output;
        }

        public unsafe float[] GetCurrentOutput()
        {
            var output = new float[DiffEqnCount];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(CollapsingInstances.Last(), true, 0, DiffEqnCount * 4, new IntPtr((void*)array));
                OpenCLCommandQueue.Finish();
            }
            return output;
        }

        public void Dispose()
        {
            DiffEquationsInstances.Dispose();
            foreach (var x in CollapsingInstances)
                x.Dispose();
            InputMulInstance.Dispose();
            OutputMulInstance.Dispose();
            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
        }
    }
}
