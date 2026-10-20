using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace GenerativeFill.Core;

public enum DeviceMode { Auto, GPU, CPU }

internal static class OrtSupport
{
    private static bool bootstrapped;

    /// <summary>Makes sure onnxruntime.dll / DirectML.dll are found next to the plugin DLL.</summary>
    public static void Bootstrap()
    {
        if (bootstrapped) return;
        bootstrapped = true;
        string dir = Path.GetDirectoryName(typeof(OrtSupport).Assembly.Location)!;

        foreach (string probe in new[] { dir, Path.Combine(dir, "runtimes", "win-x64", "native") })
        {
            string dml = Path.Combine(probe, "DirectML.dll");
            if (File.Exists(dml)) { NativeLibrary.TryLoad(dml, out _); break; }
        }

        NativeLibrary.SetDllImportResolver(typeof(OrtEnv).Assembly, (name, asm, path) =>
        {
            if (!name.StartsWith("onnxruntime", StringComparison.OrdinalIgnoreCase)) return IntPtr.Zero;
            foreach (string probe in new[] { dir, Path.Combine(dir, "runtimes", "win-x64", "native") })
            {
                string candidate = Path.Combine(probe, "onnxruntime.dll");
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr h)) return h;
            }
            return IntPtr.Zero;
        });
    }

    public static SessionOptions Options(bool gpu, int deviceId)
    {
        var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (gpu)
        {
            so.EnableMemoryPattern = false;                 // required by the DirectML EP
            so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            so.AppendExecutionProvider_DML(deviceId);
        }
        else
        {
            so.IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2);
        }
        return so;
    }

    public static NamedOnnxValue Tensor(string name, float[] data, int[] dims, TensorElementType type)
    {
        switch (type)
        {
            case TensorElementType.Float16:
                {
                    // ORT's tensor projection only accepts its own Float16 struct, not System.Half
                    // (System.Half makes OrtValue.CreateFromTensorObject throw a NullReferenceException).
                    var h = new Half[data.Length];
                    for (int i = 0; i < h.Length; i++) h[i] = (Half)data[i];
                    var f16 = MemoryMarshal.Cast<Half, Float16>(h.AsSpan()).ToArray();   // same 2-byte layout
                    return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(f16, dims));
                }
            case TensorElementType.BFloat16:
                {
                    var b = new BFloat16[data.Length];
                    var raw = MemoryMarshal.Cast<BFloat16, ushort>(b.AsSpan());
                    for (int i = 0; i < raw.Length; i++) raw[i] = (ushort)(BitConverter.SingleToInt32Bits(data[i]) >> 16);
                    return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<BFloat16>(b, dims));
                }
            case TensorElementType.Float:
                return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, dims));
            case TensorElementType.Int64:
                return NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<long>(data.Select(v => (long)v).ToArray(), dims));
            case TensorElementType.Int32:
                return NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<int>(data.Select(v => (int)v).ToArray(), dims));
            default:
                throw new NotSupportedException($"Unsupported model input type {type}");
        }
    }

    public static float[] ToFloats(DisposableNamedOnnxValue v)
    {
        // The runtime surfaces fp16 as either System.Half or Microsoft.ML.OnnxRuntime.Float16
        // depending on version; AsTensor<T>() returns null on a mismatch, so probe each.
        if (v.AsTensor<float>() is Tensor<float> tf) return tf.ToArray();

        if (v.AsTensor<Half>() is Tensor<Half> th)
        {
            Half[] h = th.ToArray();
            var f = new float[h.Length];
            for (int i = 0; i < f.Length; i++) f[i] = (float)h[i];
            return f;
        }
        if (v.AsTensor<Float16>() is Tensor<Float16> t16)
        {
            Float16[] h = t16.ToArray();
            var raw = MemoryMarshal.Cast<Float16, Half>(h.AsSpan());   // same 2-byte layout
            var f = new float[h.Length];
            for (int i = 0; i < f.Length; i++) f[i] = (float)raw[i];
            return f;
        }
        if (v.AsTensor<BFloat16>() is Tensor<BFloat16> tb)
        {
            BFloat16[] h = tb.ToArray();
            var raw = MemoryMarshal.Cast<BFloat16, ushort>(h.AsSpan());
            var f = new float[h.Length];
            for (int i = 0; i < f.Length; i++) f[i] = BitConverter.Int32BitsToSingle(raw[i] << 16);
            return f;
        }
        if (v.AsTensor<double>() is Tensor<double> td) return td.ToArray().Select(x => (float)x).ToArray();

        throw new NotSupportedException($"Unsupported model output type {v.ElementType} ({v.ElementType.GetType().Name})");
    }
}

/// <summary>
/// One ONNX session. In Auto mode a DirectML failure (at load or at run time) silently
/// falls back to a CPU session, as requested.
/// </summary>
internal sealed class OrtModel : IDisposable
{
    private readonly string path;
    private readonly string cpuPath;
    private readonly DeviceMode mode;
    private readonly int deviceId;
    private InferenceSession session;

    public bool OnGpu { get; private set; }
    public string? FallbackReason { get; private set; }
    public IReadOnlyDictionary<string, NodeMetadata> Inputs => session.InputMetadata;
    public IReadOnlyList<string> InputNames => session.InputNames;

    /// <param name="cpuPath">Optional different file (e.g. an fp32 export) to use when running on the CPU.</param>
    public OrtModel(string path, DeviceMode mode, int deviceId, string? cpuPath = null)
    {
        OrtSupport.Bootstrap();
        this.path = path; this.cpuPath = cpuPath ?? path; this.mode = mode; this.deviceId = deviceId;

        if (mode == DeviceMode.CPU) { session = Open(false); return; }
        try { session = Open(true); OnGpu = true; }
        catch (Exception ex) when (mode == DeviceMode.Auto)
        {
            FallbackReason = ex.Message;
            session = Open(false);
        }
    }

    private InferenceSession Open(bool gpu) => new(gpu ? path : cpuPath, OrtSupport.Options(gpu, deviceId));

    public TensorElementType TypeOf(string name) => session.InputMetadata[name].ElementDataType;

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        try { return session.Run(inputs); }
        catch (Exception ex) when (OnGpu && mode == DeviceMode.Auto)
        {
            FallbackReason = ex.Message;
            session.Dispose();
            session = Open(false);
            OnGpu = false;
            return session.Run(inputs);
        }
    }

    public void Dispose() => session.Dispose();
}