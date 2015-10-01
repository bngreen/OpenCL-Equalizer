using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OpenCLDSP
{
    public class CascadedIIRFilter
    {
        public IList<DifferenceEquation> Sections { get; set; }
        public float[] ScaleValues { get; set; }
        public CascadedIIRFilter(string SOSM, string scaleValues)
        {
            SOSM = SOSM.Replace("[", "").Replace("]", "");
            scaleValues = scaleValues.Replace("[", "").Replace("]", "");
            Sections = new List<DifferenceEquation>();
            foreach (var x in SOSM.Split(new string[] { ";" }, StringSplitOptions.RemoveEmptyEntries))
            {
                var vals = x.Split(new string[] { " ", "," }, StringSplitOptions.RemoveEmptyEntries);
                var coeffs = new float[6];
                for (int i = 0; i < vals.Length; i++)
                    coeffs[i] = Convert.ToSingle(vals[i]);

                var dffEqn = new DifferenceEquation()
                {
                    B = new float[4]{coeffs[0], coeffs[1], coeffs[2], 0},
                    A = new float[4]{coeffs[3], -coeffs[4], -coeffs[5], 0}
                };
                Sections.Add(dffEqn);
            }
            var sVals = scaleValues.Split(new string[] { ";" }, StringSplitOptions.RemoveEmptyEntries);
            ScaleValues = new float[sVals.Length-1];
            for (int i = 0; i < ScaleValues.Length; i++)
                ScaleValues[i] = Convert.ToSingle(sVals[i]);
        }
    }
}
