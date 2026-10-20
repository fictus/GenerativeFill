using System.Security.Cryptography;

namespace GenerativeFill.Core;

internal sealed record ModelFile(string RelPath, string Url, long ApproxBytes, string? Sha256 = null);

/// <summary>Model locations, first-use download (resumable) and optional SHA-256 verification.</summary>
internal static class ModelStore
{
    private const string Base = "https://huggingface.co/sona-forge/sd15-base-fp16/resolve/main/";
    private const string Inpaint = "https://huggingface.co/sona-forge/sd15-inpainting-fp16/resolve/main/";

    // Fill in Sha256 once you have pinned the exact files you ship; null = size/length check only.
    public static readonly ModelFile[] Sd15InpaintFp16 =
    {
        new("tokenizer/vocab.json",     Base + "tokenizer/vocab.json",     1_100_000),
        new("tokenizer/merges.txt",     Base + "tokenizer/merges.txt",       525_000),
        new("text_encoder/model.onnx",  Base + "text_encoder/model.onnx", 246_000_000),
        new("vae_encoder/model.onnx",   Base + "vae_encoder/model.onnx",   68_000_000),
        new("vae_decoder/model.onnx",   Base + "vae_decoder/model.onnx",   99_000_000),
        new("unet/model.onnx",          Inpaint + "unet/model.onnx",    1_720_220_149),
    };

    public static string Root => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "GenerativeFill", "models");

    public static string PathOf(string rel) => Path.Combine(Root, rel.Replace('/', Path.DirectorySeparatorChar));

    /// <summary>Optional LCM-LoRA-fused inpainting UNet; the user drops it here to enable Fast mode.</summary>
    public static string LcmUnet => PathOf("unet_lcm/model.onnx");
    public static bool HasLcm => File.Exists(LcmUnet);

    public static bool IsInstalled => Sd15InpaintFp16.All(f => File.Exists(PathOf(f.RelPath)));
    public static long MissingBytes => Sd15InpaintFp16.Where(f => !File.Exists(PathOf(f.RelPath))).Sum(f => f.ApproxBytes);

    private static readonly HttpClient Http = new(new HttpClientHandler { AllowAutoRedirect = true })
    { Timeout = Timeout.InfiniteTimeSpan };

    public static async Task EnsureAsync(Action<string, double> progress, CancellationToken ct)
    {
        var todo = Sd15InpaintFp16.Where(f => !File.Exists(PathOf(f.RelPath))).ToList();
        long total = todo.Sum(f => f.ApproxBytes), done = 0;

        foreach (ModelFile f in todo)
        {
            string finalPath = PathOf(f.RelPath), part = finalPath + ".part";
            Directory.CreateDirectory(Path.GetDirectoryName(finalPath)!);
            long have = File.Exists(part) ? new FileInfo(part).Length : 0;

            using var req = new HttpRequestMessage(HttpMethod.Get, f.Url);
            if (have > 0) req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(have, null);
            using HttpResponseMessage resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);

            if (resp.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
            { File.Delete(part); have = 0; throw new IOException("Partial download was invalid; please retry."); }
            resp.EnsureSuccessStatusCode();
            if (resp.StatusCode != System.Net.HttpStatusCode.PartialContent) have = 0; // server ignored Range

            await using (Stream net = await resp.Content.ReadAsStreamAsync(ct))
            await using (var fs = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write))
            {
                var buf = new byte[1 << 20];
                long fileDone = have;
                int n;
                while ((n = await net.ReadAsync(buf, ct)) > 0)
                {
                    await fs.WriteAsync(buf.AsMemory(0, n), ct);
                    fileDone += n;
                    progress($"Downloading {f.RelPath}  ({(done + fileDone) / 1048576} MB)",
                             Math.Min(1.0, (double)(done + fileDone) / total));
                }
            }

            if (f.Sha256 is { Length: > 0 })
            {
                progress($"Verifying {f.RelPath}", Math.Min(1.0, (double)done / total));
                await using var s = File.OpenRead(part);
                string hash = Convert.ToHexString(await SHA256.HashDataAsync(s, ct));
                if (!hash.Equals(f.Sha256, StringComparison.OrdinalIgnoreCase))
                { File.Delete(part); throw new InvalidDataException($"Hash mismatch for {f.RelPath}."); }
            }

            File.Move(part, finalPath, true);
            done += f.ApproxBytes;
        }
    }
}
