using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BlueWave.Interop.Asio;

namespace Asio
{
    public class Asio
    {

        AsioDriver driver;
        public static IEnumerable<InstalledDriver> InstalledDrivers { get { return AsioDriver.InstalledDrivers; } }

        public Asio(InstalledDriver drv)
        {
            ProcessAudio = null;

            // make sure we have at least one ASIO driver installed
            if (AsioDriver.InstalledDrivers.Length == 0)
            {
                return;
            }

            // load and activate the desited driver
            driver = AsioDriver.SelectDriver(drv);

            // driver.ShowControlPanel();

            driver.CreateBuffers(false);

            // this is our buffer fill event we need to respond to
            driver.BufferUpdate += new EventHandler(AsioDriver_BufferUpdate);
        }

        public void Start()
        {
            driver.Start();
        }

        public void Stop()
        {
            driver.Stop();
        }

        public delegate void ProcessAudioDelegate(IList<float> buffer, uint count);
        public ProcessAudioDelegate ProcessAudio;

        /// <summary>
        /// Called when a buffer update is required
        /// </summary>
        private void AsioDriver_BufferUpdate(object sender, EventArgs e)
        {
            // the driver is the sender
            AsioDriver driver = sender as AsioDriver;

            Channel input = driver.InputChannels[0];
            Channel leftOutput = driver.OutputChannels[0];
            Channel rightOutput = driver.OutputChannels[1];

            if (ProcessAudio != null)
            {
                ProcessAudio.Invoke(new AsioChannel(input), (uint)leftOutput.BufferSize);
            }


            for (int index = 0; index < leftOutput.BufferSize; index++)
            {
                leftOutput[index] = input[index];
                rightOutput[index] = input[index];
            }
        }
    }
}
