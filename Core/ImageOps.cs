using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace GenerativeFill.Core;

/// <summary>The square window around the selection that is sent to the model.</summary>
internal readonly record struct CropPlan(int WinX, int WinY, int Side, int SelX0, int SelY0, int SelX1, int SelY1);

internal static class ImageOps
{
    public const int ModelSize = 512;

    /// <summary>
    /// Picks a square window around the selection bounds (selection + context), clamped to the image.
    /// Selection coordinates in the plan are in image space and clipped to the window.
    /// </summary>
    public static CropPlan Plan(int imgW, int imgH, int selX, int selY, int selW, int selH, int contextPercent)
    {
        int selMax = Math.Max(selW, selH);
        int side = (int)(selMax * (1 + contextPercent / 100.0));
        side = Math.Max(side, 256);
        side = Math.Min(side, Math.Min(imgW, imgH));
        side = Math.Max(side, 8);

        int cx = selX + selW / 2, cy = selY + selH / 2;
        int wx = Math.Clamp(cx - side / 2, 0, imgW - side);
        int wy = Math.Clamp(cy - side / 2, 0, imgH - side);

        return new CropPlan(wx, wy, side,
            Math.Max(selX, wx), Math.Max(selY, wy),
            Math.Min(selX + selW, wx + side), Math.Min(selY + selH, wy + side));
    }

    /// <summary>Debug aid: writes a BGRA buffer to %LocalAppData%\GenerativeFill\debug\.</summary>
    public static void DumpPng(byte[] bgra, int w, int h, string name)
    {
        try
        {
            string dir = Path.Combine(Path.GetDirectoryName(Log.PathOfLog)!, "debug");
            Directory.CreateDirectory(dir);
            using var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            BitmapData d = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            for (int y = 0; y < h; y++) Marshal.Copy(bgra, y * w * 4, d.Scan0 + y * d.Stride, w * 4);
            bmp.UnlockBits(d);
            bmp.Save(Path.Combine(dir, name), ImageFormat.Png);
        }
        catch { }
    }

    public static byte[] Resize(byte[] bgra, int w, int h, int nw, int nh)
    {
        if (w == nw && h == nh) return (byte[])bgra.Clone();

        using var src = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        BitmapData sd = src.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        for (int y = 0; y < h; y++) Marshal.Copy(bgra, y * w * 4, sd.Scan0 + y * sd.Stride, w * 4);
        src.UnlockBits(sd);

        using var dst = new Bitmap(nw, nh, PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(dst))
        using (var attrs = new ImageAttributes())
        {
            attrs.SetWrapMode(WrapMode.TileFlipXY);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.CompositingMode = CompositingMode.SourceCopy;
            g.DrawImage(src, new Rectangle(0, 0, nw, nh), 0, 0, w, h, GraphicsUnit.Pixel, attrs);
        }

        var result = new byte[nw * nh * 4];
        BitmapData dd = dst.LockBits(new Rectangle(0, 0, nw, nh), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        for (int y = 0; y < nh; y++) Marshal.Copy(dd.Scan0 + y * dd.Stride, result, y * nw * 4, nw * 4);
        dst.UnlockBits(dd);
        return result;
    }

    /// <summary>
    /// Blends generated pixels over the original window. Alpha is 0 outside the selection box and ramps
    /// up over <paramref name="feather"/> pixels inside it, except along edges that touch the image border.
    /// </summary>
    public static byte[] Composite(byte[] original, byte[] generated, CropPlan p, int imgW, int imgH, int feather)
    {
        int s = p.Side;
        int x0 = p.SelX0 - p.WinX, x1 = p.SelX1 - p.WinX, y0 = p.SelY0 - p.WinY, y1 = p.SelY1 - p.WinY;
        bool rampL = p.SelX0 > 0, rampR = p.SelX1 < imgW, rampT = p.SelY0 > 0, rampB = p.SelY1 < imgH;
        var o = new byte[s * s * 4];

        for (int y = 0; y < s; y++)
        for (int x = 0; x < s; x++)
        {
            int i = (y * s + x) * 4;
            double a = 0;
            if (x >= x0 && x < x1 && y >= y0 && y < y1)
            {
                a = 1;
                if (feather > 0)
                {
                    if (rampL) a = Math.Min(a, (x - x0 + 0.5) / feather);
                    if (rampR) a = Math.Min(a, (x1 - x - 0.5) / feather);
                    if (rampT) a = Math.Min(a, (y - y0 + 0.5) / feather);
                    if (rampB) a = Math.Min(a, (y1 - y - 0.5) / feather);
                    a = Math.Clamp(a, 0, 1);
                    a = a * a * (3 - 2 * a); // smoothstep
                }
            }
            for (int c = 0; c < 3; c++)
                o[i + c] = (byte)Math.Round(original[i + c] * (1 - a) + generated[i + c] * a);
            o[i + 3] = (byte)Math.Round(original[i + 3] * (1 - a) + 255 * a);
        }
        return o;
    }
}
