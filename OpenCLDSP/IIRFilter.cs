using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OpenCLDSP
{
    public class IIRFilter
    {
        public IList<float> A { get; set; }
        public IList<float> B { get; set; }
        public IIRFilter(string a, string b)
        {
            b = b.Replace("[", "").Replace("]", "");
            var coeffs = b.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
            B = new List<float>();
            foreach (var x in coeffs)
                B.Add(Convert.ToSingle(x));
            a = a.Replace("[", "").Replace("]", "");
            coeffs = a.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
            A = new List<float>();
            foreach (var x in coeffs)
                A.Add(Convert.ToSingle(x));
        }
    }
}
