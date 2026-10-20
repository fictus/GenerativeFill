using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace GenerativeFill.Core;

/// <summary>CLIP BPE tokenizer (same behaviour as the Hugging Face CLIPTokenizer used by SD 1.5).</summary>
internal sealed class ClipTokenizer
{
    private const int Bos = 49406, Eos = 49407;

    private static readonly Regex Pattern = new(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private readonly Dictionary<string, int> vocab;
    private readonly Dictionary<(string, string), int> ranks = new();
    private readonly char[] byteToUnicode = new char[256];
    private readonly Dictionary<string, string> cache = new();

    public ClipTokenizer(string vocabPath, string mergesPath)
    {
        vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath))
                ?? throw new InvalidDataException("vocab.json is empty");

        int rank = 0;
        foreach (string line in File.ReadLines(mergesPath).Skip(1))
        {
            string[] p = line.Split(' ');
            if (p.Length == 2) ranks[(p[0], p[1])] = rank++;
        }

        var bs = new List<int>();
        bs.AddRange(Enumerable.Range(33, 94));    // '!'..'~'
        bs.AddRange(Enumerable.Range(161, 12));   // '¡'..'¬'
        bs.AddRange(Enumerable.Range(174, 82));   // '®'..'ÿ'
        int n = 0;
        for (int b = 0; b < 256; b++)
            byteToUnicode[b] = bs.Contains(b) ? (char)b : (char)(256 + n++);
    }

    /// <summary>Returns exactly <paramref name="maxLen"/> token ids (BOS ... EOS, then padId).</summary>
    public int[] Encode(string text, int maxLen = 77, int padId = Eos)
    {
        text = Regex.Replace(text ?? "", @"\s+", " ").Trim().ToLowerInvariant();
        var ids = new List<int> { Bos };

        foreach (Match m in Pattern.Matches(text))
        {
            var sb = new StringBuilder();
            foreach (byte b in Encoding.UTF8.GetBytes(m.Value)) sb.Append(byteToUnicode[b]);
            foreach (string piece in Bpe(sb.ToString()).Split(' '))
                if (vocab.TryGetValue(piece, out int id)) ids.Add(id);
        }

        if (ids.Count > maxLen - 1) ids.RemoveRange(maxLen - 1, ids.Count - (maxLen - 1));
        ids.Add(Eos);
        while (ids.Count < maxLen) ids.Add(padId);
        return ids.ToArray();
    }

    private string Bpe(string token)
    {
        if (cache.TryGetValue(token, out string? hit)) return hit;

        var word = token.Select(c => c.ToString()).ToList();
        word[^1] += "</w>";

        while (word.Count > 1)
        {
            int best = int.MaxValue;
            (string, string) bestPair = default;
            for (int i = 0; i < word.Count - 1; i++)
                if (ranks.TryGetValue((word[i], word[i + 1]), out int r) && r < best)
                { best = r; bestPair = (word[i], word[i + 1]); }
            if (best == int.MaxValue) break;

            var merged = new List<string>();
            for (int i = 0; i < word.Count; i++)
            {
                if (i < word.Count - 1 && word[i] == bestPair.Item1 && word[i + 1] == bestPair.Item2)
                { merged.Add(bestPair.Item1 + bestPair.Item2); i++; }
                else merged.Add(word[i]);
            }
            word = merged;
        }

        string result = string.Join(" ", word);
        cache[token] = result;
        return result;
    }
}
