using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OpenCLDSP
{
    public class FIRFilter
    {
        public IList<float> B { get; set; }
        public FIRFilter(string v)
        {
            v = v.Replace("[", "").Replace("]", "");
            var coeffs = v.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
            B = new List<float>();
            foreach (var x in coeffs)
                B.Add(Convert.ToSingle(x));
        }
    }
}
