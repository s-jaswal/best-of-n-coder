"""
best-of-n-coder — sample more, not bigger.

Samples N candidate solutions from Claude for each coding problem,
verifies each candidate against hand-written unit tests in a sandboxed
subprocess, and measures accuracy as a function of N.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python best_of_n_coder.py
"""

import os
import sys
import subprocess
import tempfile
import re
import matplotlib.pyplot as plt
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from env
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# 1. Problems — each has a prompt and a list of assert-based tests.
#    Feel free to swap these out or add your own.
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "name": "is_palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome, ignoring case and spaces.",
        "tests": [
            "assert is_palindrome('Racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('A man a plan a canal Panama'.replace(' ', '')) == True",
        ],
    },
    {
        "name": "fizzbuzz_list",
        "prompt": "Write a Python function `fizzbuzz_list(n)` that returns a list of strings for numbers 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, otherwise the number as a string.",
        "tests": [
            "assert fizzbuzz_list(5) == ['1','2','Fizz','4','Buzz']",
            "assert fizzbuzz_list(15)[14] == 'FizzBuzz'",
        ],
    },
    {
        "name": "second_largest",
        "prompt": "Write a Python function `second_largest(nums)` that returns the second largest unique value in a list of integers.",
        "tests": [
            "assert second_largest([1, 3, 3, 5, 2]) == 3",
            "assert second_largest([10, 20]) == 10",
            "assert second_largest([4, 4, 4, 8]) == 4",
        ],
    },
    {
        "name": "is_prime",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise. Handle n <= 1 correctly.",
        "tests": [
            "assert is_prime(2) == True",
            "assert is_prime(1) == False",
            "assert is_prime(17) == True",
            "assert is_prime(15) == False",
        ],
    },
    {
        "name": "matrix_transpose",
        "prompt": "Write a Python function `transpose(matrix)` that takes a 2D list (list of lists) and returns its transpose.",
        "tests": [
            "assert transpose([[1,2],[3,4]]) == [[1,3],[2,4]]",
            "assert transpose([[1,2,3]]) == [[1],[2],[3]]",
        ],
    },
    {
        "name": "balanced_parens",
        "prompt": "Write a Python function `is_balanced(s)` that returns True if all brackets in string s are balanced and correctly nested. Consider (), [], and {} as bracket pairs. Ignore all other characters.",
        "tests": [
            "assert is_balanced('([{}])') == True",
            "assert is_balanced('([)]') == False",
            "assert is_balanced('a(b)c[d]e') == True",
            "assert is_balanced('(((') == False",
            "assert is_balanced('') == True",
        ],
    },
    {
        "name": "roman_to_int",
        "prompt": "Write a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtractive notation correctly (e.g. IV = 4, IX = 9, XL = 40, CM = 900).",
        "tests": [
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('IV') == 4",
            "assert roman_to_int('LVIII') == 58",
            "assert roman_to_int('MCMXCIV') == 1994",
        ],
    },
    {
        "name": "merge_intervals",
        "prompt": "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] intervals (not necessarily sorted) and merges all overlapping intervals, returning the merged list sorted by start.",
        "tests": [
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
            "assert merge_intervals([[1,4],[0,4]]) == [[0,4]]",
        ],
    },
    {
        "name": "word_break",
        "prompt": "Write a Python function `word_break(s, word_dict)` that returns True if string s can be segmented into a space-separated sequence of one or more words from word_dict (a list of strings). Words can be reused.",
        "tests": [
            "assert word_break('leetcode', ['leet','code']) == True",
            "assert word_break('applepenapple', ['apple','pen']) == True",
            "assert word_break('catsandog', ['cats','dog','sand','and','cat']) == False",
        ],
    },
    {
        "name": "longest_common_prefix",
        "prompt": "Write a Python function `longest_common_prefix(strs)` that returns the longest common prefix string among a list of strings. Return an empty string if there is no common prefix or the list is empty.",
        "tests": [
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'",
            "assert longest_common_prefix(['dog','racecar','car']) == ''",
            "assert longest_common_prefix([]) == ''",
            "assert longest_common_prefix(['single']) == 'single'",
        ],
    },
]

# ---------------------------------------------------------------------------
# 2. Sampling — call Claude to generate one candidate solution.
# ---------------------------------------------------------------------------

def extract_code(text):
    """Pull code out of a ```python ... ``` block if present, else return as-is."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def get_candidate(prompt):
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nRespond with ONLY the function definition in a single Python code block. No explanation.",
        }],
    )
    text = response.content[0].text
    return extract_code(text)


# ---------------------------------------------------------------------------
# 3. Verifier — run candidate code + a test line in an isolated subprocess.
# ---------------------------------------------------------------------------

def run_test(code, test_line, timeout=5):
    script = code + "\n\n" + test_line + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.remove(path)


def passes_all_tests(code, tests):
    return all(run_test(code, t) for t in tests)


# ---------------------------------------------------------------------------
# 4. Solve — try up to N candidates, return True on first pass (+ transcript).
# ---------------------------------------------------------------------------

def solve(problem, n):
    for i in range(n):
        code = get_candidate(problem["prompt"])
        if passes_all_tests(code, problem["tests"]):
            return True, i + 1, code
    return False, n, code  # last attempt kept for transcript purposes


# ---------------------------------------------------------------------------
# 5. Run the sweep across N values and collect results.
# ---------------------------------------------------------------------------

def main():
    n_values = [1, 3, 10]
    results = {n: 0 for n in n_values}
    transcripts = []  # store one interesting example: solved at N>1 but not N=1

    for problem in PROBLEMS:
        print(f"\n--- {problem['name']} ---")
        first_attempt_code = None
        solved_at = None

        for n in n_values:
            solved, attempts_used, code = solve(problem, n)
            if solved:
                results[n] += 1
            print(f"  N={n}: {'PASS' if solved else 'FAIL'} (attempts used: {attempts_used})")

            if n == 1:
                first_attempt_code = code
            if solved and solved_at is None:
                solved_at = (n, code)

        # capture a transcript where N=1 failed but a higher N succeeded
        if solved_at and solved_at[0] > 1:
            transcripts.append({
                "problem": problem["name"],
                "n1_code": first_attempt_code,
                "solved_n": solved_at[0],
                "solved_code": solved_at[1],
            })

    total = len(PROBLEMS)
    print("\n=== Summary ===")
    for n in n_values:
        pct = results[n] / total * 100
        print(f"N={n}: {results[n]}/{total} solved ({pct:.0f}%)")

    # ---- chart ----
    accuracies = [results[n] / total * 100 for n in n_values]
    plt.figure(figsize=(6, 4))
    plt.plot(n_values, accuracies, marker="o", linewidth=2)
    plt.xlabel("N (samples per problem)")
    plt.ylabel("% problems solved")
    plt.title("best-of-n-coder: accuracy vs. N")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.savefig("accuracy_vs_n.png", dpi=150, bbox_inches="tight")
    print("\nChart saved to accuracy_vs_n.png")

    # ---- print one transcript for the README ----
    if transcripts:
        t = transcripts[0]
        print(f"\n=== Example transcript: {t['problem']} ===")
        print(f"N=1 attempt (failed):\n{t['n1_code']}\n")
        print(f"Solved at N={t['solved_n']}:\n{t['solved_code']}\n")
    else:
        print("\nNo N=1-fails-but-higher-N-succeeds example found this run — "
              "try rerunning, sampling is stochastic.")


if __name__ == "__main__":
    main()