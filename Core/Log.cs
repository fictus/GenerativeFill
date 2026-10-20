namespace GenerativeFill.Core;

internal static class Log
{
    public static string PathOfLog => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "GenerativeFill", "log.txt");

    public static void Write(string message)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(PathOfLog)!);
            File.AppendAllText(PathOfLog, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}  {message}{Environment.NewLine}");
        }
        catch { /* logging must never break the effect */ }
    }
}
