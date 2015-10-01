using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BlueWave.Interop.Asio;

namespace Asio
{
    class AsioChannel : IList<float>
    {
        Channel channel;
        public AsioChannel(Channel cn)
        {
            channel = cn;
        }


        public int IndexOf(float item)
        {
            throw new NotImplementedException();
        }

        public void Insert(int index, float item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public float this[int index]
        {
            get
            {
                return channel[index];
            }
            set
            {
                channel[index] = value;
            }
        }

        public void Add(float item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public bool Contains(float item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(float[] array, int arrayIndex)
        {
            for (int i = 0; i < Count; i++)
                array[i] = this[i];
        }

        public int Count
        {
            get { return channel.BufferSize; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(float item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<float> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}
