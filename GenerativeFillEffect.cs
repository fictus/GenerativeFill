using GenerativeFill.Core;
using PaintDotNet;
using PaintDotNet.Effects;
using PaintDotNet.Imaging;
using PaintDotNet.IndirectUI;
using PaintDotNet.PropertySystem;
using PaintDotNet.Rendering;
using System.Windows.Forms;

namespace GenerativeFill;

public sealed class PluginSupportInfo : IPluginSupportInfo
{
    public string DisplayName => "Generative Fill";
    public string Author => "Luis";
    public string Copyright => "";
    public Version Version => typeof(PluginSupportInfo).Assembly.GetName().Version!;
    public Uri WebsiteUri => new("https://www.getpaint.net/");
}

[PluginSupportInfo<PluginSupportInfo>(DisplayName = "Generative Fill")]
public sealed class GenerativeFillEffect : PaintDotNet.Effects.BitmapEffect
{
    private static readonly GenSettings Last = new();

    private readonly object gate = new();
    private bool initialized;
    private byte[]? composite;
    private CropPlan plan;

    // valid only while the prompt dialog is open
    private IEffectInputBitmap<ColorBgra32>? activeSrc;
    private int imgW, imgH;
    private RectInt32 sel;

    public GenerativeFillEffect()
        : base("Generative Fill", (IBitmapSource?)null, "Artistic", BitmapEffectOptions.Create() with { IsConfigurable = false })
    {
    }

    /// <summary>Opens the prompt window once per effect run (first tile to render wins; others wait).</summary>
    private void EnsureDialog(IEffectInputBitmap<ColorBgra32> src)
    {
        lock (gate)
        {
            if (initialized) return;
            initialized = true;
            try
            {
                activeSrc = src;
                imgW = src.Size.Width; imgH = src.Size.Height;
                sel = Environment.Selection.RenderBounds;
                GenerativeFill.Core.Log.Write($"opening dialog; image {imgW}x{imgH}, selection {sel.X},{sel.Y} {sel.Width}x{sel.Height}");

                GenResult? r = PromptDialog.Show(Last, Generate);
                // Apply in the prompt dialog is the only confirmation: render() below paints the result and
                // Paint.NET commits it immediately (the effect is not configurable, so no second OK dialog).
                if (r != null) { composite = r.Composite; plan = r.Plan; GenerativeFill.Core.Log.Write("applied"); }
                else GenerativeFill.Core.Log.Write("dialog closed without applying");
            }
            catch (Exception ex)
            {
                GenerativeFill.Core.Log.Write("ERROR: " + ex);
                MessageBox.Show("Generative Fill failed:\n\n" + ex.Message + "\n\nDetails: " + GenerativeFill.Core.Log.PathOfLog,
                    "Generative Fill", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally { activeSrc = null; }
        }
    }

    /// <summary>Runs on a background thread while the dialog is open.</summary>
    private GenResult Generate(GenSettings s, Action<string, double> report, CancellationToken ct)
    {
        IEffectInputBitmap<ColorBgra32> src = activeSrc ?? throw new InvalidOperationException("Source image not available.");
        if (sel.Width <= 0 || sel.Height <= 0) throw new InvalidOperationException("Select an area of the image first.");

        GenerativeFill.Core.Log.Write($"generate: prompt='{s.Prompt}' seed={s.Seed} preset={s.Preset} device={s.Device}");

        if (!ModelStore.IsInstalled)
            ModelStore.EnsureAsync(report, ct).GetAwaiter().GetResult();
        ct.ThrowIfCancellationRequested();

        CropPlan cp = ImageOps.Plan(imgW, imgH, sel.X, sel.Y, sel.Width, sel.Height, s.Context);
        int S = cp.Side;

        var orig = new byte[S * S * 4];
        using (IBitmapLock<ColorBgra32> l = src.Lock(new RectInt32(0, 0, imgW, imgH)))
        {
            RegionPtr<ColorBgra32> reg = l.AsRegionPtr();
            for (int y = 0; y < S; y++)
                for (int x = 0; x < S; x++)
                {
                    ColorBgra32 c = reg[cp.WinX + x, cp.WinY + y];
                    int i = (y * S + x) * 4;
                    orig[i] = c.B; orig[i + 1] = c.G; orig[i + 2] = c.R; orig[i + 3] = c.A;
                }
        }

        const int M = ImageOps.ModelSize;
        var req = new InpaintRequest
        {
            Prompt = s.Prompt,
            Negative = s.Negative,
            Seed = s.Seed,
            Size = M,
            Preset = s.Preset,
            Device = s.Device,
            DmlDeviceId = s.GpuIndex,
            Guidance = s.Guidance,
            Strength = s.Strength,
            ImageBgra = ImageOps.Resize(orig, S, S, M, M),
        };

        // model mask = selection box grown by 8 px
        const int grow = 8;
        double k = (double)M / S;
        int mx0 = Math.Max(0, (int)Math.Floor((cp.SelX0 - cp.WinX) * k) - grow);
        int my0 = Math.Max(0, (int)Math.Floor((cp.SelY0 - cp.WinY) * k) - grow);
        int mx1 = Math.Min(M, (int)Math.Ceiling((cp.SelX1 - cp.WinX) * k) + grow);
        int my1 = Math.Min(M, (int)Math.Ceiling((cp.SelY1 - cp.WinY) * k) + grow);
        req.Mask = new byte[M * M];
        for (int y = my0; y < my1; y++) Array.Fill(req.Mask, (byte)255, y * M + mx0, mx1 - mx0);

        byte[] gen = InpaintPipeline.Run(req, report, ct);
        ImageOps.DumpPng(req.ImageBgra, M, M, "debug_input.png");
        ImageOps.DumpPng(gen, M, M, "debug_generated.png");

        // how much did the model actually change inside the mask?
        long diff = 0, cnt = 0;
        for (int p = 0; p < M * M; p++)
            if (req.Mask[p] != 0)
                for (int c = 0; c < 3; c++) { diff += Math.Abs(gen[p * 4 + c] - req.ImageBgra[p * 4 + c]); cnt++; }
        double mad = cnt > 0 ? (double)diff / cnt : 0;
        GenerativeFill.Core.Log.Write($"mean change inside mask: {mad:0.0}/255");
        byte[] genFull = ImageOps.Resize(gen, M, M, S, S);
        GenerativeFill.Core.Log.Write("generation finished");

        return new GenResult
        {
            Composite = ImageOps.Composite(orig, genFull, cp, imgW, imgH, s.Feather),
            Plan = cp,
            Note = mad < 6 ? "The result is almost identical to the original. Try a new seed, a more specific prompt, or a bigger selection." : "",
        };
    }

    protected override void OnRender(IBitmapEffectOutput output)
    {
        using IEffectInputBitmap<ColorBgra32> src = Environment.GetSourceBitmapBgra32();
        EnsureDialog(src);

        using IBitmapLock<ColorBgra32> srcLock = src.Lock(new RectInt32(0, 0, src.Size.Width, src.Size.Height));
        RegionPtr<ColorBgra32> s = srcLock.AsRegionPtr();
        RectInt32 b = output.Bounds;
        using IBitmapLock<ColorBgra32> dstLock = output.LockBgra32();
        RegionPtr<ColorBgra32> d = dstLock.AsRegionPtr();

        byte[]? comp = composite;
        CropPlan p = plan;

        for (int y = b.Top; y < b.Bottom; y++)
            for (int x = b.Left; x < b.Right; x++)
            {
                ColorBgra32 c = s[x, y];
                if (comp != null && x >= p.WinX && x < p.WinX + p.Side && y >= p.WinY && y < p.WinY + p.Side)
                {
                    int i = ((y - p.WinY) * p.Side + (x - p.WinX)) * 4;
                    c = ColorBgra32.FromBgra(comp[i], comp[i + 1], comp[i + 2], comp[i + 3]);
                }
                d[x - b.Left, y - b.Top] = c;   // output buffer is tile-relative, source is image-relative
            }
    }
}
