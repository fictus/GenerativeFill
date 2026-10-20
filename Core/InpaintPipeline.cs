using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace GenerativeFill.Core;

public enum Preset { Auto, FastLCM, Speed, Balanced, Quality }

internal sealed class InpaintRequest
{
    public string Prompt = "";
    public string Negative = "";
    public int Seed;
    public Preset Preset = Preset.Auto;
    public DeviceMode Device = DeviceMode.Auto;
    public int DmlDeviceId = 0;
    public double Guidance = 7.5;
    public double Strength = 1.0;
    public int Size = 512;
    /// <summary>Size x Size BGRA crop of the image.</summary>
    public byte[] ImageBgra = Array.Empty<byte>();
    /// <summary>Size x Size, 255 where the model must generate.</summary>
    public byte[] Mask = Array.Empty<byte>();
}

internal static class InpaintPipeline
{
    private const int TokenLen = 77;

    /// <summary>Returns a Size x Size BGRA image with the generated content.</summary>
    public static byte[] Run(InpaintRequest r, Action<string, double> report, CancellationToken ct)
    {
        int S = r.Size, L = S / 8;
        const string dir = "";
        const float vaeScale = 0.18215f;
        const int hidden = 768;

        Log.Write($"pipeline: size={S} device={r.Device} preset={r.Preset} cfg={r.Guidance}");
        var tok = new ClipTokenizer(ModelStore.PathOf(dir + "tokenizer/vocab.json"), ModelStore.PathOf(dir + "tokenizer/merges.txt"));

        // ---- 1. text embeddings -------------------------------------------------------------
        report("Encoding prompt…", 0.02);
        float[] condEmb, uncondEmb;
        using (var te = new OrtModel(ModelStore.PathOf("text_encoder/model.onnx"), r.Device, r.DmlDeviceId))
        {
            condEmb = Embed(te, tok, r.Prompt);
            uncondEmb = Embed(te, tok, r.Negative);
        }
        ct.ThrowIfCancellationRequested();

        // ---- 2. image / masked-image latents -------------------------------------------------
        report("Encoding image…", 0.06);
        var img = new float[3 * S * S];          // CHW, RGB, [-1,1]
        var masked = new float[3 * S * S];
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
            {
                int p = y * S + x;
                bool fill = r.Mask[p] >= 128;
                for (int c = 0; c < 3; c++)
                {
                    float v = r.ImageBgra[p * 4 + (2 - c)] / 127.5f - 1f;
                    img[c * S * S + p] = v;
                    masked[c * S * S + p] = fill ? 0f : v;
                }
            }

        float[] maskedLatent, imageLatent = Array.Empty<float>();
        using (var ve = new OrtModel(ModelStore.PathOf(dir + "vae_encoder/model.onnx"), r.Device, r.DmlDeviceId))
        {
            maskedLatent = Scale(EncodeVae(ve, masked, S), vaeScale);
            if (r.Strength < 0.999) imageLatent = Scale(EncodeVae(ve, img, S), vaeScale);
        }
        ct.ThrowIfCancellationRequested();

        var maskLatent = new float[L * L];       // nearest-neighbour downsample, as diffusers does
        for (int y = 0; y < L; y++)
            for (int x = 0; x < L; x++)
                maskLatent[y * L + x] = r.Mask[(y * 8 + 4) * S + x * 8 + 4] >= 128 ? 1f : 0f;

        // ---- 3. denoising loop ---------------------------------------------------------------
        report("Loading model…", 0.10);
        string unetPath = ModelStore.PathOf(dir + "unet/model.onnx");
        bool wantLcm = (r.Preset == Preset.FastLCM || (r.Preset == Preset.Auto && r.Device == DeviceMode.CPU && ModelStore.HasLcm));
        if (r.Preset == Preset.FastLCM && !ModelStore.HasLcm)
            throw new FileNotFoundException("Fast (LCM) needs an LCM-fused inpainting UNet at: " + ModelStore.LcmUnet);
        if (wantLcm) unetPath = ModelStore.LcmUnet;

        using var unet = new OrtModel(unetPath, r.Device, r.DmlDeviceId);
        string nSample = Pick(unet, "sample", 0), nTime = Pick(unet, "timestep", 1), nEnc = Pick(unet, "encoder_hidden_states", 2);
        int? chans = unet.Inputs[nSample].Dimensions.Length > 1 ? unet.Inputs[nSample].Dimensions[1] : null;
        if (chans is > 0 && chans != 9)   // -1 = dynamic axis, unknown until run time
            throw new InvalidDataException($"UNet expects {chans} input channels; an inpainting UNet (9 channels) is required.");

        Preset eff = r.Preset;
        if (eff == Preset.Auto) eff = wantLcm ? Preset.FastLCM : (unet.OnGpu ? Preset.Balanced : Preset.Speed);
        IScheduler sched = eff switch
        {
            Preset.FastLCM => new LcmScheduler(4),
            Preset.Speed => new DdimScheduler(12),
            Preset.Quality => new DdimScheduler(30),
            _ => new DdimScheduler(20),
        };
        bool lcm = sched is LcmScheduler;
        bool cfg = !lcm && r.Guidance > 1.0;
        int batch = cfg ? 2 : 1;

        var rng = new Random(r.Seed);
        var noise = new float[4 * L * L];
        for (int i = 0; i < noise.Length; i++) noise[i] = Sched.Randn(rng);

        int first = 0;
        float[] latents = noise;
        if (r.Strength < 0.999 && !lcm)
        {
            first = Math.Clamp(sched.Timesteps.Length - (int)(sched.Timesteps.Length * r.Strength), 0, sched.Timesteps.Length - 1);
            latents = sched.AddNoise(imageLatent, noise, sched.Timesteps[first]);
        }

        // embeddings: [uncond; cond] when using classifier-free guidance
        float[] emb = cfg ? uncondEmb.Concat(condEmb).ToArray() : condEmb;
        int plane = 4 * L * L;

        for (int i = first; i < sched.Timesteps.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            double frac = 0.12 + 0.78 * (i - first) / Math.Max(1, sched.Timesteps.Length - first);
            report($"Step {i - first + 1}/{sched.Timesteps.Length - first}  ({(unet.OnGpu ? "GPU" : "CPU")}, {eff})", frac);

            var input = new float[batch * 9 * L * L];
            for (int b = 0; b < batch; b++)
            {
                int o = b * 9 * L * L;
                Array.Copy(latents, 0, input, o, plane);
                Array.Copy(maskLatent, 0, input, o + plane, L * L);
                Array.Copy(maskedLatent, 0, input, o + plane + L * L, plane);
            }

            var feeds = new List<NamedOnnxValue>
            {
                OrtSupport.Tensor(nSample, input, new[] { batch, 9, L, L }, unet.TypeOf(nSample)),
                OrtSupport.Tensor(nTime, new float[] { sched.Timesteps[i] }, new[] { 1 }, unet.TypeOf(nTime)),
                OrtSupport.Tensor(nEnc, emb, new[] { batch, TokenLen, hidden }, unet.TypeOf(nEnc)),
            };

            float[] eps;
            using (var res = unet.Run(feeds))
            {
                float[] outp = OrtSupport.ToFloats(res.First());
                if (cfg)
                {
                    eps = new float[plane];
                    float g = (float)r.Guidance;
                    for (int k = 0; k < plane; k++) eps[k] = outp[k] + g * (outp[plane + k] - outp[k]);
                }
                else eps = outp;
            }
            latents = sched.Step(eps, i, latents, rng);
        }
        Log.Write($"unet done: onGpu={unet.OnGpu} fallback={unet.FallbackReason ?? "none"}");
        unet.Dispose();

        // ---- 4. decode -----------------------------------------------------------------------
        report("Decoding…", 0.93);
        ct.ThrowIfCancellationRequested();
        float[] rgb;
        using (var vd = new OrtModel(ModelStore.PathOf(dir + "vae_decoder/model.onnx"), r.Device, r.DmlDeviceId))
        {
            string nIn = Pick(vd, "latent_sample", 0);
            using var res = vd.Run(new[]
            {
                OrtSupport.Tensor(nIn, Scale(latents, 1f / vaeScale), new[] { 1, 4, L, L }, vd.TypeOf(nIn)),
            });
            rgb = OrtSupport.ToFloats(res.First());
        }

        var bgra = new byte[S * S * 4];
        for (int p = 0; p < S * S; p++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = rgb[c * S * S + p] / 2f + 0.5f;
                if (float.IsNaN(v)) v = 0.5f;
                v = Math.Clamp(v, 0f, 1f);
                bgra[p * 4 + (2 - c)] = (byte)Math.Round(v * 255f);
            }
            bgra[p * 4 + 3] = 255;
        }
        return bgra;
    }

    private static string Pick(OrtModel m, string preferred, int index) =>
        m.Inputs.ContainsKey(preferred) ? preferred : m.InputNames[index];

    private static float[] Scale(float[] a, float k)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] * k;
        return r;
    }

    private static float[] Embed(OrtModel te, ClipTokenizer tok, string text)
    {
        string name = te.InputNames[0];
        float[] ids = tok.Encode(text, TokenLen).Select(i => (float)i).ToArray();
        using var res = te.Run(new[] { OrtSupport.Tensor(name, ids, new[] { 1, TokenLen }, te.TypeOf(name)) });
        return OrtSupport.ToFloats(res.First());   // last_hidden_state [1,77,768]
    }

    private static float[] EncodeVae(OrtModel ve, float[] chw, int size)
    {
        string name = ve.InputNames[0];
        using var res = ve.Run(new[] { OrtSupport.Tensor(name, chw, new[] { 1, 3, size, size }, ve.TypeOf(name)) });
        float[] all = OrtSupport.ToFloats(res.First());   // [1,4,L,L] sample (or [1,8,L,L] mean+logvar)
        int plane = 4 * (size / 8) * (size / 8);
        return all.Length > plane ? all[..plane] : all;    // first 4 channels = latent / mean
    }
}
