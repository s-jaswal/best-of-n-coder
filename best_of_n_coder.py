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
        "name": "flatten",
        "prompt": "Write a Python function `flatten(nested)` that flattens an arbitrarily nested list of integers into a single flat list.",
        "tests": [
            "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]",
            "assert flatten([]) == []",
            "assert flatten([[1], [2], [3]]) == [1, 2, 3]",
        ],
    },
    {
        "name": "most_frequent_char",
        "prompt": "Write a Python function `most_frequent_char(s)` that returns the most frequently occurring character in string s. Break ties by returning the character that appears first in the string.",
        "tests": [
            "assert most_frequent_char('aabbbcc') == 'b'",
            "assert most_frequent_char('xyz') == 'x'",
        ],
    },
    {
        "name": "reverse_words",
        "prompt": "Write a Python function `reverse_words(sentence)` that reverses the order of words in a sentence, keeping each word itself unreversed. Words are separated by single spaces.",
        "tests": [
            "assert reverse_words('the sky is blue') == 'blue is sky the'",
            "assert reverse_words('hello world') == 'world hello'",
        ],
    },
    {
        "name": "sum_digits",
        "prompt": "Write a Python function `sum_digits(n)` that returns the sum of the digits of a non-negative integer n.",
        "tests": [
            "assert sum_digits(1234) == 10",
            "assert sum_digits(0) == 0",
            "assert sum_digits(9) == 9",
        ],
    },
    {
        "name": "unique_chars",
        "prompt": "Write a Python function `has_unique_chars(s)` that returns True if all characters in string s are unique (no repeats), False otherwise.",
        "tests": [
            "assert has_unique_chars('abcdef') == True",
            "assert has_unique_chars('hello') == False",
            "assert has_unique_chars('') == True",
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