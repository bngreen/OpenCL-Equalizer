using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using OpenCLDSP;
using Asio;
using System.Threading;
using NAudio.Wave;


namespace Equalizer
{
    public partial class Form1 : Form
    {
        FIRSolver2 solver;
        Asio.Asio asio;
        float[] table;
        WaveFileWriter writer;
        bool save = true;
        public Form1()
        {
            if (save)
                writer = new WaveFileWriter("test.wav", new WaveFormat(44100, 1));
            Control.CheckForIllegalCrossThreadCalls = false;
            //35
            //40-60
            //60-100
            //100-300
            //600-900
            //900-1800
            //1800-3500
            //3500-7500
            //7500-12000
            //12000-18000
            //18000
            var filters = new EqualizerFilters();
            table = new float[filters.Filters.Count];
            for (int i = 0; i < table.Length; i++)
                table[i] = 1;
            solver = new FIRSolver2(OpenCLNet.OpenCL.GetPlatform(0), filters.Filters, 2048);
            solver.SetMulTable(table);

            InitializeComponent();

            asio = new Asio.Asio(Asio.Asio.InstalledDrivers.ElementAt(0));
            asio.ProcessAudio = ProcessAudio;
            var asioThread = new Thread(new ThreadStart(asio.Start));
            asioThread.IsBackground = true;
            asioThread.Priority = ThreadPriority.Highest;
            asioThread.Start();
        }
        float max = 0;
        void ProcessAudio(IList<float> buffer, uint frames)
        {
            if (true)
            {
                var volume = ((float)trackBar13.Value) / 100;
                var buff = buffer.ToArray();
                var buff2 = solver.PerformAndSum(buff);

                for (int i = 0; i < buffer.Count; i++)
                {
                    buffer[i] = volume * buff2[i];
                    max = Math.Max(buff2[i], max);
                }
                if (save)
                    writer.WriteSamples(buff2, 0, buff2.Length);
            }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            //asio.Stop();
            if (save)
                writer.Close();
        }

        void updateTable(int index, int value)
        {
            table[index] = ((float)value) / 100;
            solver.SetMulTable(table);
        }

        private void trackBar1_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(0, trackBar1.Value);
        }

        private void trackBar1_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(0, trackBar1.Value);
        }

        private void trackBar2_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(1, trackBar2.Value);
        }

        private void trackBar2_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(1, trackBar2.Value);
        }

        private void trackBar3_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(2, trackBar3.Value);
        }

        private void trackBar3_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(2, trackBar3.Value);
        }

        private void trackBar4_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(3, trackBar4.Value);
        }

        private void trackBar4_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(3, trackBar4.Value);
        }

        private void trackBar5_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(4, trackBar5.Value);
        }

        private void trackBar5_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(4, trackBar5.Value);
        }

        private void trackBar6_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(5, trackBar6.Value);
        }

        private void trackBar6_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(5, trackBar6.Value);
        }

        private void trackBar7_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(6, trackBar7.Value);
        }

        private void trackBar7_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(6, trackBar7.Value);
        }

        private void trackBar8_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(7, trackBar8.Value);
        }

        private void trackBar8_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(7, trackBar8.Value);
        }

        private void trackBar9_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(8, trackBar9.Value);
        }

        private void trackBar9_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(8, trackBar9.Value);
        }

        private void trackBar10_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(9, trackBar10.Value);
        }

        private void trackBar10_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(9, trackBar10.Value);
        }

        private void trackBar11_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(10, trackBar11.Value);
        }

        private void trackBar11_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(10, trackBar11.Value);
        }

        private void trackBar12_MouseUp(object sender, MouseEventArgs e)
        {
            updateTable(11, trackBar12.Value);
        }

        private void trackBar12_KeyUp(object sender, KeyEventArgs e)
        {
            updateTable(11, trackBar12.Value);
        }
    }
}
