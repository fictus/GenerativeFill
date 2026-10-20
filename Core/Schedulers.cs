namespace GenerativeFill.Core;

internal static class Sched
{
    /// <summary>SD 1.5 "scaled_linear" cumulative alphas (1000 train steps).</summary>
    public static double[] AlphasCumprod()
    {
        const int n = 1000;
        var acp = new double[n];
        double s0 = Math.Sqrt(0.00085), s1 = Math.Sqrt(0.012), cum = 1;
        for (int i = 0; i < n; i++)
        {
            double b = s0 + (s1 - s0) * i / (n - 1);
            cum *= 1 - b * b;
            acp[i] = cum;
        }
        return acp;
    }

    public static float Randn(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}

internal interface IScheduler
{
    int[] Timesteps { get; }
    float[] AddNoise(float[] x0, float[] noise, int timestep);
    float[] Step(float[] modelOutput, int index, float[] sample, Random rng);
}

/// <summary>DDIM, eta = 0, "leading" spacing with steps_offset = 1 (the SD 1.5 default).</summary>
internal sealed class DdimScheduler : IScheduler
{
    private readonly double[] acp = Sched.AlphasCumprod();
    private readonly int ratio;
    public int[] Timesteps { get; }

    public DdimScheduler(int steps)
    {
        ratio = Math.Max(1, 1000 / steps);
        Timesteps = Enumerable.Range(0, steps).Select(i => i * ratio + 1).Reverse().ToArray();
    }

    public float[] AddNoise(float[] x0, float[] noise, int t)
    {
        double a = Math.Sqrt(acp[t]), b = Math.Sqrt(1 - acp[t]);
        var r = new float[x0.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(a * x0[i] + b * noise[i]);
        return r;
    }

    public float[] Step(float[] eps, int index, float[] x, Random rng)
    {
        int t = Timesteps[index], prev = t - ratio;
        double a = acp[t], ap = prev >= 0 ? acp[prev] : acp[0];
        double sa = Math.Sqrt(a), sb = Math.Sqrt(1 - a), sap = Math.Sqrt(ap), sbp = Math.Sqrt(1 - ap);
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++)
        {
            double x0 = (x[i] - sb * eps[i]) / sa;
            r[i] = (float)(sap * x0 + sbp * eps[i]);
        }
        return r;
    }
}

/// <summary>LCM multistep sampler. Only valid with an LCM-LoRA-fused UNet.</summary>
internal sealed class LcmScheduler : IScheduler
{
    private readonly double[] acp = Sched.AlphasCumprod();
    public int[] Timesteps { get; }

    public LcmScheduler(int steps)
    {
        const int originalSteps = 50, c = 1000 / originalSteps;
        var origin = Enumerable.Range(1, originalSteps).Select(i => i * c - 1).ToArray(); // 19..999
        int skip = origin.Length / steps;
        Timesteps = Enumerable.Range(0, steps).Select(i => origin[origin.Length - 1 - i * skip]).ToArray();
    }

    public float[] AddNoise(float[] x0, float[] noise, int t)
    {
        double a = Math.Sqrt(acp[t]), b = Math.Sqrt(1 - acp[t]);
        var r = new float[x0.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(a * x0[i] + b * noise[i]);
        return r;
    }

    public float[] Step(float[] model, int index, float[] x, Random rng)
    {
        int t = Timesteps[index];
        bool last = index == Timesteps.Length - 1;
        int prevT = last ? t : Timesteps[index + 1];
        double a = acp[t], ap = prevT >= 0 ? acp[prevT] : 1.0;

        const double sigmaData = 0.5;
        double scaled = t * 10.0;
        double cSkip = sigmaData * sigmaData / (scaled * scaled + sigmaData * sigmaData);
        double cOut = scaled / Math.Sqrt(scaled * scaled + sigmaData * sigmaData);

        var r = new float[x.Length];
        double sa = Math.Sqrt(a), sb = Math.Sqrt(1 - a), sap = Math.Sqrt(ap), sbp = Math.Sqrt(1 - ap);
        for (int i = 0; i < r.Length; i++)
        {
            double pred = (x[i] - sb * model[i]) / sa;
            double denoised = cOut * pred + cSkip * x[i];
            r[i] = last ? (float)denoised : (float)(sap * denoised + sbp * Sched.Randn(rng));
        }
        return r;
    }
}
