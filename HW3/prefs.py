# align/prefs.py
import re

# Returns the length of the longest run of identical characters in text
def max_run(text: str) -> int:
    if not text: return 0
    best, cur, prev = 1, 1, text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = ch
    return best

# Computes a reward based on the density of 's' and 'S' characters in the text,
# penalizing for long runs of identical characters.
def s_density_reward(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return 0.0
    s_count = sum(1 for ch in letters if ch in ("s","S"))
    s_density = s_count / len(letters)
    rep_pen = min(0.3, max_run(text) / 50.0)  # cap penalty
    r = s_density * (1.0 - rep_pen)
    return max(0.0, min(1.0, r))
