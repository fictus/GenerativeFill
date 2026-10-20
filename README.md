# Generative Fill for Paint.NET

A Paint.NET 5.x effect plugin that fills a selected area of your image with AI-generated content from a text prompt. It runs **locally** using Stable Diffusion 1.5 Inpainting through ONNX Runtime (DirectML GPU acceleration, with automatic CPU fallback). No cloud service or account is needed.

- **Menu location:** `Effects > Artistic > Generative Fill`
- **Platform:** Windows x64 only
- **Version:** 0.1.0

---

## Requirements

| Requirement | Details |
|---|---|
| OS | Windows 10/11, 64-bit |
| Paint.NET | 5.x (the version that ships with .NET 9), installed on disk (the classic/Store builds may use different plugin folders) |
| .NET SDK | **.NET 9 SDK** (only needed to build from source) |
| Disk space | About **2.2 GB** free for the model download |
| Internet | Needed once, for the first-use model download |
| GPU (optional) | Any DirectX 12 GPU speeds things up a lot. Without one, the plugin falls back to the CPU (works, but is slow) |

---

## Installation

### 1. Build the plugin

1. Install the [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0).
2. Unzip `GenerativeFillEffect.zip` somewhere, e.g. `C:\dev\GenerativeFillEffect`.
3. Open a terminal in that folder (the one containing `GenerativeFill.csproj`) and run:

   ```powershell
   dotnet build GenerativeFill.csproj -c Release
   ```

   The project looks for the Paint.NET assemblies (`PaintDotNet.*.dll`) in `C:\Program Files\paint.net\`. If Paint.NET is installed somewhere else, point the build at it. Keep the trailing backslash:

   ```powershell
   dotnet build GenerativeFill.csproj -c Release -p:PdnDir="D:\Apps\paint.net\"
   ```

4. When it finishes, the output is in:

   ```
   bin\Release\net9.0-windows\win-x64\
   ```

### 2. Copy the files into Paint.NET

1. **Close Paint.NET** completely.
2. Open your Paint.NET `Effects` folder. For a standard install this is:

   ```
   C:\Program Files\paint.net\Effects\
   ```

   (Writing here requires administrator rights. Accept the Windows UAC prompt when copying.)
3. Copy the **entire contents** of the build output folder (`bin\Release\net9.0-windows\win-x64\`) into that `Effects` folder. This must include:
   - `GenerativeFill.dll`
   - `Microsoft.ML.OnnxRuntime.dll`
   - `onnxruntime.dll` and `DirectML.dll` (either directly in the folder, or under `runtimes\win-x64\native\`, so keep that subfolder if it exists)
   - `System.Drawing.Common.dll` and any other `.dll` files in the folder

   > Do **not** copy any `PaintDotNet.*.dll` files, since Paint.NET already provides those. They are referenced but intentionally not copied by the build.

   > The plugin looks for `onnxruntime.dll` and `DirectML.dll` next to `GenerativeFill.dll`, or in `runtimes\win-x64\native\` beside it. If the dependency DLLs are missing, the effect will fail with an error.

4. Start Paint.NET. If Windows blocks the DLLs ("This file came from another computer"), right-click each DLL, choose **Properties**, tick **Unblock**, and restart Paint.NET.

### 3. Verify

Open any image in Paint.NET and check for **Effects > Artistic > Generative Fill**. If it isn't there, see [Troubleshooting](#troubleshooting).

---

## First-run model download

The first time you click **Generate**, the plugin asks to download the Stable Diffusion 1.5 Inpainting model (about 2.2 GB, split across several files: tokenizer, text encoder, VAE encoder/decoder and the UNet).

- Models are stored in `%LocalAppData%\GenerativeFill\models\`
- Downloads are **resumable**. If your connection drops, click Generate again and it continues from the partial file.
- The model is distributed under the **CreativeML Open RAIL-M** license, which includes use restrictions. Review it before use.
- This only happens once. Afterwards the plugin works fully offline.

---

## How to use

1. Open an image in Paint.NET.
2. **Select the area you want to fill** using any selection tool (Rectangle Select works well). The plugin uses the selection's bounding box, so a rough box around the target area is enough.
3. Choose **Effects > Artistic > Generative Fill**. The prompt window opens.
4. Type what you want in the **Prompt** box, e.g. `a wooden bench under a tree, soft daylight`.
5. Click **Generate**. A progress bar shows each step, and the status line says whether it is running on GPU or CPU.
6. A preview of the area around your selection appears on the right.
   - Not happy? Click **Generate** again for a new variation (a new random seed is used each time by default).
   - Happy? Click **Apply**.
7. Paint.NET applies the result **immediately**. There is no second OK dialog. Use `Ctrl+Z` to undo if needed.

Click **Cancel** to close without changing anything. While a generation is running, the button becomes **Stop** and cancels it.

### Settings reference

| Setting | What it does |
|---|---|
| **Prompt** | What to generate inside the selection. Be specific (subject, style, lighting). |
| **Negative prompt** | What to avoid. Default: `blurry, low quality, distorted, watermark, text`. |
| **Seed** | Number that controls the randomness. The same seed and settings give the same result. |
| **New random seed each time** | When on, every re-generate picks a new seed. Turn it off to keep the seed fixed while you tweak other settings. |
| **Quality / speed** | `Auto`, `FastLCM`, `Speed` (12 steps), `Balanced` (20 steps), `Quality` (30 steps). Auto picks Balanced on GPU and Speed on CPU. |
| **Device** | `Auto` (GPU, falling back to CPU on failure), `GPU` or `CPU`. |
| **GPU adapter index** | Which GPU to use (0 = first). Change this on multi-GPU systems. |
| **Prompt strength (CFG)** | How strongly the result follows your prompt (1–15, default 9). Higher is more literal. |
| **Change strength** | 0.30–1.00 (default 1.00). Lower values keep more of the original content under the selection. |
| **Edge feather (px)** | Softness of the blend between generated and original pixels (0–64, default 8). |
| **Context around selection (%)** | How much surrounding image the model sees (25–300%, default 40). More context helps the fill match its surroundings. |

Your settings are remembered for the rest of the Paint.NET session.

### Tips for better results

- **Describe the content, not the action.** Write `a red brick wall with ivy` instead of `remove the person`.
- **Use a bigger selection** if the result looks almost unchanged. The plugin will tell you when the output is nearly identical to the original.
- **Match the surroundings.** Mention lighting and style (`overcast light, photo` or `watercolor painting`) so the fill blends in.
- **Increase context** if the fill looks pasted-in, and **raise feather** to soften visible seams.
- The model works at **512×512**. The area around your selection is scaled to that size and back, so very large selections will look softer. For big images, fill in smaller pieces.
- Try several seeds. Results vary a lot from one to the next.

---

## Optional: Fast (LCM) mode

For much faster CPU generation (4 steps), you can supply an LCM-LoRA-fused **inpainting** UNet in ONNX format:

```
%LocalAppData%\GenerativeFill\models\unet_lcm\model.onnx
```

With that file present, choose **FastLCM** in *Quality / speed*. `Auto` will also use it when running on the CPU. This file is **not** downloaded automatically, and the UNet must have 9 input channels (inpainting). Choosing FastLCM without it shows an error that names the expected path.

---

## Troubleshooting

**The effect doesn't appear in the menu**
- Make sure the DLLs are in the correct `Effects` folder and Paint.NET was fully restarted.
- Confirm your Paint.NET is 5.x and that you built against that same installation (`-p:PdnDir=...`).
- Unblock the DLLs (right-click > Properties > Unblock).

**Build error: cannot find `PaintDotNet.*.dll`**
- Pass the correct install folder with `-p:PdnDir="<path>\"`, including the trailing backslash.

**"Select an area of the image first."**
- You need an active selection before running the effect.

**It's very slow**
- You're probably running on the CPU. Check the status line during generation. Set **Device** to `GPU` (and try a different **GPU adapter index**) or install the optional LCM model. Lower the **Quality / speed** preset to `Speed`.

**Out-of-memory or GPU errors**
- In `Auto` mode the plugin silently falls back to the CPU. If you forced `GPU`, switch to `Auto` or `CPU`. Update your graphics drivers.

**Download fails or stalls**
- Click Generate again to resume. Check that your firewall allows access to `huggingface.co`.

**"Hash mismatch" / corrupted model**
- Delete the affected file under `%LocalAppData%\GenerativeFill\models\` and click Generate to download it again.

**Result looks identical to the original**
- Try a new seed, a more specific prompt, a larger selection, and make sure **Change strength** is 1.00.

### Log and debug files

- Log: `%LocalAppData%\GenerativeFill\log.txt`. Include it when reporting a problem.
- Debug images (the model input and raw output of the last run): `%LocalAppData%\GenerativeFill\debug\`

---

## Uninstall

1. Close Paint.NET.
2. Delete `GenerativeFill.dll` and the dependency DLLs you copied from the `Effects` folder.
3. Optionally delete `%LocalAppData%\GenerativeFill\` to remove the models (about 2.2 GB), logs and settings.

---

## Project layout

```
GenerativeFill.csproj          Project file (net9.0-windows, win-x64)
GenerativeFillEffect.cs        Paint.NET effect entry point
Core/
  PromptDialog.cs              Prompt / settings / preview window
  InpaintPipeline.cs           Stable Diffusion inpainting pipeline (ONNX Runtime)
  ModelStore.cs                Model locations and resumable download
  OrtSupport.cs                ONNX Runtime setup, DirectML with CPU fallback
  Schedulers.cs                DDIM and LCM schedulers
  ClipTokenizer.cs             CLIP text tokenizer
  ImageOps.cs                  Crop planning, resizing, feathered compositing
  Log.cs                       Logging
```

## Credits and licenses

- Stable Diffusion 1.5 Inpainting weights are subject to the CreativeML Open RAIL-M license.
- Inference is powered by [ONNX Runtime](https://onnxruntime.ai/) with DirectML.
- Built for [Paint.NET](https://www.getpaint.net/).
