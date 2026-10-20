using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace GenerativeFill.Core;

internal sealed class GenSettings
{
    public string Prompt = "";
    public string Negative = "blurry, low quality, distorted, watermark, text";
    public int Seed = Random.Shared.Next(0, 1_000_000);
    public bool RandomSeedEachTime = true;
    public Preset Preset = Preset.Auto;
    public DeviceMode Device = DeviceMode.Auto;
    public int GpuIndex = 0;
    public double Guidance = 9.0;
    public double Strength = 1.0;
    public int Feather = 8;
    public int Context = 40;

    public GenSettings Clone() => (GenSettings)MemberwiseClone();
}

internal sealed class GenResult
{
    public byte[] Composite = Array.Empty<byte>();   // BGRA, Plan.Side x Plan.Side
    public CropPlan Plan;
    public string Note = "";
}

internal delegate GenResult GenerateFunc(GenSettings settings, Action<string, double> report, CancellationToken ct);

/// <summary>Prompt window: settings, a Generate button, a preview and Apply / Cancel.</summary>
internal sealed class PromptDialog : Form
{
    private readonly GenSettings settings;
    private readonly GenerateFunc generate;

    private readonly TextBox prompt = new() { Multiline = true, ScrollBars = ScrollBars.Vertical };
    private readonly TextBox negative = new() { Multiline = true, ScrollBars = ScrollBars.Vertical };
    private readonly NumericUpDown seed = new() { Minimum = 0, Maximum = int.MaxValue };
    private readonly CheckBox randomSeed = new() { Text = "New random seed each time", AutoSize = true };
    private readonly ComboBox preset = new() { DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly ComboBox device = new() { DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly NumericUpDown gpu = new() { Minimum = 0, Maximum = 7 };
    private readonly NumericUpDown guidance = new() { Minimum = 1, Maximum = 15, DecimalPlaces = 1, Increment = 0.5M };
    private readonly NumericUpDown strength = new() { Minimum = 0.3M, Maximum = 1, DecimalPlaces = 2, Increment = 0.05M };
    private readonly NumericUpDown feather = new() { Minimum = 0, Maximum = 64 };
    private readonly NumericUpDown context = new() { Minimum = 25, Maximum = 300 };
    private readonly Button btnGenerate = new() { Text = "Generate", Width = 110, Height = 32 };
    private readonly Button btnApply = new() { Text = "Apply", Width = 90, Height = 32, Enabled = false };
    private readonly Button btnCancel = new() { Text = "Cancel", Width = 90, Height = 32 };
    private readonly Label status = new() { AutoSize = false, Height = 20, Text = "Type a prompt and click Generate." };
    private readonly ProgressBar bar = new() { Minimum = 0, Maximum = 1000, Height = 16 };
    private readonly PictureBox preview = new() { SizeMode = PictureBoxSizeMode.Zoom, BorderStyle = BorderStyle.FixedSingle, BackColor = Color.FromArgb(32, 32, 32) };

    private CancellationTokenSource? cts;
    private bool busy;
    private GenResult? current;

    /// <summary>The result the user accepted with Apply, or null.</summary>
    public GenResult? Applied { get; private set; }

    private PromptDialog(GenSettings s, GenerateFunc gen)
    {
        settings = s; generate = gen;

        Text = "Generative Fill";
        ClientSize = new Size(700, 560);
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false; MinimizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        TopMost = true;

        const int lx = 12, cx = 170, w = 300;
        int y = 12;
        void Row(string label, Control c, int h = 24, int width = w)
        {
            Controls.Add(new Label { Text = label, Left = lx, Top = y + 3, Width = cx - lx - 6 });
            c.Left = cx; c.Top = y; c.Width = width; c.Height = h;
            Controls.Add(c);
            y += h + 8;
        }

        Row("Prompt", prompt, 80);
        Row("Negative prompt", negative, 50);
        Row("Seed", seed, 24, 120);
        randomSeed.Left = cx; randomSeed.Top = y - 2; Controls.Add(randomSeed); y += 26;
        Row("Quality / speed", preset);
        Row("Device", device);
        Row("GPU adapter index", gpu, 24, 80);
        Row("Prompt strength (CFG)", guidance, 24, 80);
        Row("Change strength", strength, 24, 80);
        Row("Edge feather (px)", feather, 24, 80);
        Row("Context around selection (%)", context, 24, 80);

        preset.DataSource = Enum.GetValues<Preset>();
        device.DataSource = Enum.GetValues<DeviceMode>();

        preview.SetBounds(490, 12, 198, 198);
        Controls.Add(preview);
        Controls.Add(new Label
        {
            Left = 490, Top = 214, Width = 198, Height = 60,
            Text = "Preview of the area around your selection. Paint.NET keeps only the selected part when you click Apply.",
        });

        status.SetBounds(12, 478, 676, 20);
        bar.SetBounds(12, 502, 676, 16);
        btnGenerate.Location = new Point(12, 522);
        btnApply.Location = new Point(132, 522);
        btnCancel.Location = new Point(228, 522);
        Controls.AddRange(new Control[] { status, bar, btnGenerate, btnApply, btnCancel });
        ClientSize = new Size(700, 566);

        AcceptButton = null;
        btnGenerate.Click += async (_, _) => await OnGenerate();
        btnApply.Click += (_, _) => { Applied = current; DialogResult = DialogResult.OK; };
        btnCancel.Click += (_, _) =>
        {
            if (busy) { cts?.Cancel(); status.Text = "Cancelling…"; }
            else DialogResult = DialogResult.Cancel;
        };
        FormClosing += (_, e) =>
        {
            if (busy) { e.Cancel = true; cts?.Cancel(); status.Text = "Cancelling…"; }
        };

        ToSettings(s);
    }

    private void ToSettings(GenSettings s)
    {
        prompt.Text = s.Prompt; negative.Text = s.Negative;
        seed.Value = s.Seed; randomSeed.Checked = s.RandomSeedEachTime;
        preset.SelectedItem = s.Preset; device.SelectedItem = s.Device;
        gpu.Value = s.GpuIndex; guidance.Value = (decimal)s.Guidance; strength.Value = (decimal)s.Strength;
        feather.Value = s.Feather; context.Value = s.Context;
    }

    private void FromControls()
    {
        settings.Prompt = prompt.Text.Trim(); settings.Negative = negative.Text;
        settings.Seed = (int)seed.Value; settings.RandomSeedEachTime = randomSeed.Checked;
        settings.Preset = (Preset)preset.SelectedItem!; settings.Device = (DeviceMode)device.SelectedItem!;
        settings.GpuIndex = (int)gpu.Value; settings.Guidance = (double)guidance.Value; settings.Strength = (double)strength.Value;
        settings.Feather = (int)feather.Value; settings.Context = (int)context.Value;
    }

    private void SetBusy(bool b)
    {
        busy = b;
        foreach (Control c in Controls)
            if (c != btnCancel && c != status && c != bar && c != preview) c.Enabled = !b;
        btnApply.Enabled = !b && current != null;
        btnCancel.Text = b ? "Stop" : "Cancel";
    }

    private void Report(string text, double fraction)
    {
        if (IsDisposed || !IsHandleCreated) return;
        try
        {
            BeginInvoke(() =>
            {
                status.Text = text;
                bar.Value = (int)Math.Clamp(fraction * 1000, 0, 1000);
            });
        }
        catch { }
    }

    private async Task OnGenerate()
    {
        FromControls();
        if (settings.Prompt.Length == 0)
        {
            MessageBox.Show(this, "Type a prompt first.", "Generative Fill", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        if (!ModelStore.IsInstalled)
        {
            double gb = ModelStore.MissingBytes / 1_073_741_824.0;
            var ok = MessageBox.Show(this,
                $"Generative Fill needs to download the Stable Diffusion 1.5 Inpainting model (about {gb:0.0} GB).\n\n" +
                $"It will be stored in:\n{ModelStore.Root}\n\n" +
                "The model is distributed under the CreativeML Open RAIL-M license, which includes use restrictions.\n\nDownload now?",
                "Generative Fill", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (ok != DialogResult.Yes) return;
        }

        if (settings.RandomSeedEachTime && current != null)
        {
            settings.Seed = Random.Shared.Next(0, int.MaxValue);
            seed.Value = settings.Seed;
        }

        GenSettings snapshot = settings.Clone();
        cts = new CancellationTokenSource();
        SetBusy(true);
        bar.Value = 0;
        try
        {
            CancellationToken ct = cts.Token;
            GenResult r = await Task.Run(() => generate(snapshot, Report, ct), ct);
            current = r;
            ShowPreview(r);
            status.Text = "Done. Click Apply to keep it, or Generate again for another result." + (r.Note.Length > 0 ? "  " + r.Note : "");
            bar.Value = 1000;
        }
        catch (OperationCanceledException) { status.Text = "Cancelled."; bar.Value = 0; }
        catch (Exception ex)
        {
            Log.Write("ERROR: " + ex);
            status.Text = "Failed.";
            MessageBox.Show(this, "Generative Fill failed:\n\n" + ex.Message + "\n\nDetails: " + Log.PathOfLog,
                "Generative Fill", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            cts.Dispose(); cts = null;
            SetBusy(false);
        }
    }

    private void ShowPreview(GenResult r)
    {
        int s = r.Plan.Side;
        var bmp = new Bitmap(s, s, PixelFormat.Format32bppArgb);
        BitmapData d = bmp.LockBits(new Rectangle(0, 0, s, s), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        for (int y = 0; y < s; y++) Marshal.Copy(r.Composite, y * s * 4, d.Scan0 + y * d.Stride, s * 4);
        bmp.UnlockBits(d);
        Image? old = preview.Image;
        preview.Image = bmp;
        old?.Dispose();
    }

    /// <summary>Shows the dialog on its own STA thread and blocks until it closes.</summary>
    public static GenResult? Show(GenSettings settings, GenerateFunc generate)
    {
        GenResult? result = null;
        var t = new Thread(() =>
        {
            using var f = new PromptDialog(settings, generate);
            f.ShowDialog();
            result = f.Applied;
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();
        return result;
    }
}
