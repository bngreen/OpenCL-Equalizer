using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLNet;
using System.IO;

namespace OpenCLDSP
{
    public class CascadedIIRFilterSolver
    {
        private Platform Platform { get; set; }
        private Context OpenCLContext { get; set; }
        private CommandQueue OpenCLCommandQueue { get; set; }
        public IList<Device> OpenCLDevices { get; set; }
        private Program DiffEqnProgram { get; set; }
        private Kernel initDiffEqn { get; set; }
        private Kernel performDiffEqn { get; set; }
        private Kernel updateDiffEqn { get; set; }
        private Kernel performDiffEqn2 { get; set; }
        private Kernel updateDiffEqn2 { get; set; }
        private Kernel sumReduce { get; set; }
        private Kernel constMultiply { get; set; }

        private IList<Mem> DiffEqnInst { get; set; }
        private IList<Mem> InMulInst { get; set; }
        private IList<Mem> OutMulInst { get; set; }
        private IList<Mem> ScaleInst { get; set; }
        private Mem TempBuffer { get; set; }

        
        private IList<Mem> CollapsingInstances { get; set; }

        public int DiffEqnOrder { get; private set; }
        public int FilterCount { get; private set; }
        public int FilterSectionsCount { get; private set; }
        public unsafe CascadedIIRFilterSolver(Platform platform, IList<CascadedIIRFilter> filters)
        {

            FilterSectionsCount = filters[0].Sections.Count;
            var diffEqnOrder = 4;
            FilterCount = filters.Count;
            DiffEqnOrder = diffEqnOrder;
            var exponent = (int)Math.Round(Math.Log(diffEqnOrder, 2));

            var inMuls = new List<float[]>();
            var outMuls = new List<float[]>();
            var scales = new List<float[]>();
            for (int i = 0; i < FilterSectionsCount; i++)
            {
                var cIM = new float[filters.Count * 4];
                var cOM = new float[filters.Count * 4];
                var scale = new float[filters.Count];
                inMuls.Add(cIM);
                outMuls.Add(cOM);
                scales.Add(scale);
                for (int o = 0; o < filters.Count; o++)
                {
                    var b = filters[o].Sections[i].A[0];
                    cIM[o * 4] = filters[o].Sections[i].B[0] / b;
                    cIM[o * 4 + 1] = filters[o].Sections[i].B[1] / b;
                    cIM[o * 4 + 2] = filters[o].Sections[i].B[2] / b;
                    cIM[o * 4 + 3] = filters[o].Sections[i].B[3] / b;
                    cOM[o * 4] = 1;
                    cOM[o * 4 + 1] = filters[o].Sections[i].A[1] / b;
                    cOM[o * 4 + 2] = filters[o].Sections[i].A[2] / b;
                    cOM[o * 4 + 3] = filters[o].Sections[i].A[3] / b;
                    scale[o] = filters[o].ScaleValues[i];
                }
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
            performDiffEqn2 = DiffEqnProgram.CreateKernel("performDiffEqn2");
            updateDiffEqn2 = DiffEqnProgram.CreateKernel("updateDiffEqn2");
            sumReduce = DiffEqnProgram.CreateKernel("sumReduce");
            constMultiply = DiffEqnProgram.CreateKernel("constMultiply");

            


            DiffEqnInst = new List<Mem>();
            for (int i = 0; i < FilterSectionsCount; i++)
                DiffEqnInst.Add(OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, (4 + 8 * diffEqnOrder)*FilterCount));

            foreach (var x in DiffEqnInst)
            {
                initDiffEqn.SetArg(0, x);
                OpenCLCommandQueue.EnqueueNDRangeKernel(initDiffEqn, 1, null, new int[1] { FilterCount }, null);
                OpenCLCommandQueue.Finish();
            }

            CollapsingInstances = new List<Mem>();
            for (int i = exponent; i >= 0; i--)
                CollapsingInstances.Add(OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4*(int)Math.Pow(2, i)*FilterCount));

            InMulInst = new List<Mem>();
            OutMulInst = new List<Mem>();
            for (int i = 0; i < inMuls.Count; i++)
            {
                InMulInst.Add(OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * inMuls[i].Length));
                OutMulInst.Add(OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * outMuls[i].Length));
                fixed (float* array1 = inMuls[i], array2 = outMuls[i])
                {
                    OpenCLCommandQueue.EnqueueWriteBuffer(InMulInst[i], true, 0, 4 * inMuls[i].Length, new IntPtr((void*)array1));
                    OpenCLCommandQueue.EnqueueWriteBuffer(OutMulInst[i], true, 0, 4 * outMuls[i].Length, new IntPtr((void*)array2));
                    OpenCLCommandQueue.Finish();
                }
            }

            ScaleInst = new List<Mem>();
            foreach (var x in scales)
            {
                var m = OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * x.Length);
                fixed (float* array = x)
                {
                    OpenCLCommandQueue.EnqueueWriteBuffer(m, true, 0, 4 * x.Length, new IntPtr((void*)array));
                    OpenCLCommandQueue.Finish();
                }
                ScaleInst.Add(m);
            }

            TempBuffer = OpenCLContext.CreateBuffer(MemFlags.READ_WRITE, 4 * filters.Count);

        }
        int[] localWorkSize = null;
        public unsafe void Perform(float input)
        {
            performDiffEqn.SetArg(0, DiffEqnInst[0]);
            performDiffEqn.SetArg(1, InMulInst[0]);
            performDiffEqn.SetArg(2, OutMulInst[0]);
            performDiffEqn.SetArg(3, input);
            performDiffEqn.SetArg(4, CollapsingInstances[0]);
            OpenCLCommandQueue.EnqueueNDRangeKernel(performDiffEqn, 1, null, new int[1] { FilterCount * DiffEqnOrder }, localWorkSize);
            OpenCLCommandQueue.Finish();
            for (int i = 0; i < CollapsingInstances.Count - 1; i++)
            {
                sumReduce.SetArg(0, CollapsingInstances[i]);
                sumReduce.SetArg(1, CollapsingInstances[i + 1]);
                var size = (int)Math.Pow(2, CollapsingInstances.Count - i - 2);
                OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduce, 1, null, new int[1] { size * FilterCount }, localWorkSize);
                OpenCLCommandQueue.Finish();
            }
            updateDiffEqn.SetArg(0, DiffEqnInst[0]);
            updateDiffEqn.SetArg(1, CollapsingInstances.Last());
            updateDiffEqn.SetArg(2, input);
            OpenCLCommandQueue.EnqueueNDRangeKernel(updateDiffEqn, 1, null, new int[1] { FilterCount }, localWorkSize);
            OpenCLCommandQueue.Finish();

            constMultiply.SetArg(0, CollapsingInstances.Last());
            constMultiply.SetArg(1, ScaleInst[0]);
            constMultiply.SetArg(2, TempBuffer);
            OpenCLCommandQueue.EnqueueNDRangeKernel(constMultiply, 1, null, new int[1] { FilterCount }, localWorkSize);
            OpenCLCommandQueue.Finish();

            for (int i = 1; i < DiffEqnInst.Count; i++)
            {
                performDiffEqn2.SetArg(0, DiffEqnInst[i]);
                performDiffEqn2.SetArg(1, InMulInst[i]);
                performDiffEqn2.SetArg(2, OutMulInst[i]);
                performDiffEqn2.SetArg(3, TempBuffer);
                performDiffEqn2.SetArg(4, CollapsingInstances[0]);
                OpenCLCommandQueue.EnqueueNDRangeKernel(performDiffEqn2, 1, null, new int[1] { FilterCount * DiffEqnOrder }, localWorkSize);
                OpenCLCommandQueue.Finish();
                for (int o = 0; o < CollapsingInstances.Count - 1; o++)
                {
                    sumReduce.SetArg(0, CollapsingInstances[o]);
                    sumReduce.SetArg(1, CollapsingInstances[o + 1]);
                    var size = (int)Math.Pow(2, CollapsingInstances.Count - i - 2);
                    OpenCLCommandQueue.EnqueueNDRangeKernel(sumReduce, 1, null, new int[1] { size * FilterCount }, localWorkSize);
                    OpenCLCommandQueue.Finish();
                }

                updateDiffEqn2.SetArg(0, DiffEqnInst[i]);
                updateDiffEqn2.SetArg(1, CollapsingInstances.Last());
                updateDiffEqn2.SetArg(2, TempBuffer);
                OpenCLCommandQueue.EnqueueNDRangeKernel(updateDiffEqn2, 1, null, new int[1] { FilterCount }, localWorkSize);
                OpenCLCommandQueue.Finish();

                constMultiply.SetArg(0, CollapsingInstances.Last());
                constMultiply.SetArg(1, ScaleInst[i]);
                constMultiply.SetArg(2, TempBuffer);
                OpenCLCommandQueue.EnqueueNDRangeKernel(constMultiply, 1, null, new int[1] { FilterCount }, localWorkSize);
                OpenCLCommandQueue.Finish();
            }

        }

        public unsafe float[] GetCurrentOutput()
        {
            var output = new float[FilterCount];
            fixed (float* array = output)
            {
                OpenCLCommandQueue.EnqueueReadBuffer(TempBuffer, true, 0, FilterCount * 4, new IntPtr((void*)array));
                OpenCLCommandQueue.Finish();
            }
            return output;
        }

        public void Dispose()
        {
            foreach (var x in DiffEqnInst)
                x.Dispose();
            foreach (var x in CollapsingInstances)
                x.Dispose();
            foreach (var x in InMulInst)
                x.Dispose();
            foreach (var x in OutMulInst)
                x.Dispose();
            foreach (var x in ScaleInst)
                x.Dispose();

            OpenCLCommandQueue.Dispose();
            OpenCLContext.Dispose();
        }
    }
}
