#!/usr/bin/env python3
"""Build the WSP regression datasets and record their hashes.

Writes datasets/wsp-coding.json, datasets/wsp-general.json, datasets/wsp-format.json and
configs/wsp-regressions.json. Output is deterministic: the same inputs give the same bytes.

Usage:
    python scripts/build_wsp_regressions.py [--root DIR]

Every coding expected answer is checked by running its reference snippet in a fresh Python
subprocess. Every general answer is checked for a single correct letter. Every format verifier is
checked against a passing reference response and a failing counterexample. All content here is
original to this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LETTERS = "ABCD"

CODING_PURPOSE = (
    "Original exact-answer Python questions for WSP regression checks. Each expected answer was verified "
    "by executing its reference snippet in a fresh Python subprocess (scripts/build_wsp_regressions.py). "
    "The model answers with one JSON object and never executes the snippet."
)
GENERAL_PURPOSE = (
    "Original multiple-choice items (four options, one correct letter) covering arithmetic, logic, "
    "science facts, and reading comprehension. Arithmetic answers were recomputed by the builder."
)
FORMAT_PURPOSE = (
    "Original instruction-following items. Each item carries a programmatic verifier spec; "
    "scripts/build_wsp_regressions.py checks every verifier against a passing and a failing response."
)

# (code, question, expected). The reference snippet must assign its answer to `result`.
CODING = [
    ("x = [3, 1, 2]\nx.sort(reverse=True)\nresult = x[0] * 10 + x[-1]", "What is the value of result?", 31),
    ("text = 'Python'\nresult = text[1:4].upper()", "What is the string value of result?", "YTH"),
    ("result = len('hello, world')", "What is the value of result?", 12),
    ("result = 7 // 2 + 7 % 2 * 10", "What is the value of result?", 13),
    ("result = -7 // 2", "What is the value of result?", -4),
    ("result = -7 % 3", "What is the value of result?", 2),
    ("result = round(2.5) + round(3.5)", "What is the value of result?", 6),
    ("result = 0.1 + 0.2 == 0.3", "What is the Boolean value of result?", False),
    ("result = 2 ** -1", "What is the value of result?", 0.5),
    ("result = 5 & 3 | 8", "What is the value of result?", 9),
    ("result = 1 << 4 >> 2", "What is the value of result?", 4),
    ("def f(items=[]):\n    items.append(1)\n    return items\nf()\nresult = f()",
     "What is the value of result?", [1, 1]),
    ("funcs = [lambda: i for i in range(3)]\nresult = [f() for f in funcs]",
     "What is the value of result?", [2, 2, 2]),
    ("funcs = [lambda i=i: i for i in range(3)]\nresult = [f() for f in funcs]",
     "What is the value of result?", [0, 1, 2]),
    ("a = [1, 2]\nb = a\nb += [3]\nresult = a", "What is the value of result?", [1, 2, 3]),
    ("a = (1, 2)\nb = a\nb += (3,)\nresult = a", "What is the value of result?", [1, 2]),
    ("s = {1, 2, 3}\nt = {3, 4}\nresult = len(s ^ t)", "What is the value of result?", 3),
    ("d = {'a': 1}\nd.setdefault('a', 5)\nd.setdefault('b', 6)\nresult = d",
     "What is the value of result?", {"a": 1, "b": 6}),
    ("words = ['apple', 'Banana', 'cherry']\nresult = sorted(words, key=str.lower)",
     "What is the value of result?", ["apple", "Banana", "cherry"]),
    ("result = sorted(['b', 'A', 'a', 'B'])", "What is the value of result?", ["A", "B", "a", "b"]),
    ("nums = [1, 2, 3, 4, 5, 6]\nresult = [n for n in nums if n % 2 == 0 and n > 2]",
     "What is the value of result?", [4, 6]),
    ("result = sorted({n % 3 for n in range(10)})", "What is the value of result?", [0, 1, 2]),
    ("result = 'a,b,,c'.split(',')", "What is the value of result?", ["a", "b", "", "c"]),
    ("result = '  padded  '.strip().center(10, '*')", "What is the string value of result?", "**padded**"),
    ("result = 'abcabc'.rfind('b')", "What is the value of result?", 4),
    ("result = 'mississippi'.count('ss')", "What is the value of result?", 2),
    ("result = int('0x1f', 16)", "What is the value of result?", 31),
    ("result = int('012')", "What is the value of result?", 12),
    ("result = bool('False')", "What is the Boolean value of result?", True),
    ("matrix = [[1, 2], [3, 4]]\nresult = [list(row) for row in zip(*matrix)]",
     "What is the value of result?", [[1, 3], [2, 4]]),
    ("def gen():\n    yield 1\n    yield 2\n    return 3\n\ng = gen()\n"
     "result = list(g) + [next(g, 'done')]", "What is the value of result?", [1, 2, "done"]),
    ("class Counter:\n    count = 0\n    def __init__(self):\n        Counter.count += 1\n\n"
     "Counter()\nCounter()\nresult = Counter.count", "What is the value of result?", 2),
    ("class A:\n    def who(self):\n        return 'A'\n\nclass B(A):\n"
     "    def who(self):\n        return 'B+' + super().who()\n\nresult = B().who()",
     "What is the string value of result?", "B+A"),
    ("try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    result = 'caught'\n"
     "finally:\n    result = result + '!'", "What is the string value of result?", "caught!"),
    ("try:\n    raise KeyError('k')\nexcept LookupError as exc:\n    result = type(exc).__name__",
     "What is the string value of result?", "KeyError"),
    ("data = {'x': [1, 2], 'y': {'z': 3}}\nresult = data['y']['z'] + len(data['x'])",
     "What is the value of result?", 5),
    ("import copy\noriginal = [[1], [2]]\nshallow = copy.copy(original)\nshallow[0].append(9)\nresult = original",
     "What is the value of result?", [[1, 9], [2]]),
    ("from functools import reduce\nresult = reduce(lambda acc, n: acc * 10 + n, [1, 2, 3], 0)",
     "What is the value of result?", 123),
    ("result = max('apple', 'banana', key=len)", "What is the string value of result?", "banana"),
    ("counter = 0\nwhile True:\n    counter += 3\n    if counter > 10:\n        break\nresult = counter",
     "What is the value of result?", 12),
]

ARITHMETIC = [
    ("What is 48 + 67?", "48 + 67"),
    ("What is 1000 - 387?", "1000 - 387"),
    ("What is 14 times 16?", "14 * 16"),
    ("What is 144 divided by 12?", "144 / 12"),
    ("What is 25% of 360?", "0.25 * 360"),
    ("What is 15 squared?", "15 ** 2"),
    ("What is 2 to the power of 10?", "2 ** 10"),
    ("What is the sum of the first 10 positive integers?", "sum(range(1, 11))"),
    ("What is 7 times 8, plus 9?", "7 * 8 + 9"),
    ("What is 360 divided by 15, plus 4?", "360 / 15 + 4"),
    ("What is the least common multiple of 6 and 8?", "6 * 8 // 2"),
    ("What is three quarters of 92?", "0.75 * 92"),
    ("How many minutes are in 3.5 hours?", "3.5 * 60"),
    ("What is 999 + 1?", "999 + 1"),
    ("What is the remainder when 97 is divided by 8?", "97 % 8"),
]

LOGIC = [
    ("Every ticket in the box is blue. Ana took one ticket out of the box. Which must be true?",
     "Ana's ticket is blue", ["Ana's ticket is red", "The box is now empty", "Ana took a green ticket"]),
    ("If it rains, the match is cancelled. The match was not cancelled. Which conclusion is valid?",
     "It did not rain", ["It rained", "The match was moved indoors", "It is raining now"]),
    ("Three friends sit in seats 1, 2 and 3 from left to right. Ola is not in seat 1 or seat 3. "
     "Ben sits to the left of Ola. Cy sits in seat 3. Who sits in seat 2?",
     "Ola", ["Ben", "Cy", "Nobody"]),
    ("What number comes next in the sequence 2, 6, 12, 20, 30, ?", "42", ["40", "36", "44"]),
    ("The statement 'All swans are white' is false. Which statement is guaranteed to be true?",
     "At least one swan is not white", ["No swan is white", "All swans are black", "Every swan is white"]),
    ("Dana is taller than Eli. Eli is taller than Fay. Who is the shortest?",
     "Fay", ["Dana", "Eli", "This cannot be determined"]),
    ("Let p be true and q be false. Which compound statement is true?",
     "p or q", ["p and q", "not p", "q and not q"]),
    ("In a race, Kim finished ahead of Lee, and Lee finished ahead of Max. Which statement is true?",
     "Kim finished ahead of Max", ["Max finished ahead of Kim", "Lee finished last", "Kim finished last"]),
    ("A coin is flipped three times. How many outcomes contain exactly two heads?",
     "3", ["2", "4", "1"]),
    ("What letter comes next in the sequence A, C, E, G, ?", "I", ["H", "J", "K"]),
    ("Some cats are grey. All grey animals are fast. Which conclusion is certain?",
     "Some cats are fast", ["All cats are fast", "No cats are fast", "All cats are grey"]),
    ("What is the next term in the sequence 1, 1, 2, 3, 5, 8, ?", "13", ["11", "12", "16"]),
    ("A box has 3 red and 5 blue marbles. What is the fewest draws that guarantee at least one red marble?",
     "6", ["4", "5", "8"]),
    ("No reptiles are mammals. All snakes are reptiles. Which conclusion follows?",
     "No snakes are mammals", ["All snakes are mammals", "Some mammals are snakes", "All mammals are snakes"]),
    ("Which is the only even prime number?", "2", ["4", "6", "1"]),
]

SCIENCE = [
    ("Which gas makes up the largest share of Earth's atmosphere by volume?",
     "Nitrogen", ["Oxygen", "Carbon dioxide", "Argon"]),
    ("What is the chemical symbol for sodium?", "Na", ["So", "Sd", "Sn"]),
    ("At sea level, at what temperature in degrees Celsius does pure water boil?",
     "100", ["90", "212", "0"]),
    ("Which organelle is the main site of photosynthesis in plant cells?",
     "Chloroplast", ["Mitochondrion", "Ribosome", "Nucleus"]),
    ("What force keeps the Moon in orbit around the Earth?", "Gravity", ["Friction", "Magnetism", "Buoyancy"]),
    ("How many bones are in a typical adult human skeleton?", "206", ["180", "212", "300"]),
    ("Which planet has the most prominent ring system?", "Saturn", ["Mars", "Mercury", "Venus"]),
    ("What type of rock forms from cooled magma or lava?", "Igneous", ["Sedimentary", "Metamorphic", "Fossil"]),
    ("Which particle has a negative electric charge?", "Electron", ["Proton", "Neutron", "Photon"]),
    ("What is the SI unit of electrical resistance?", "Ohm", ["Ampere", "Volt", "Watt"]),
    ("Which blood cells are mainly responsible for fighting infection?",
     "White blood cells", ["Red blood cells", "Platelets", "Plasma"]),
    ("What is the approximate speed of light in a vacuum?",
     "300,000 km per second", ["3,000 km per second", "30,000 km per hour", "300 km per second"]),
    ("Which element has the atomic number 1?", "Hydrogen", ["Helium", "Lithium", "Carbon"]),
    ("Plants absorb which gas from the air for photosynthesis?",
     "Carbon dioxide", ["Nitrogen", "Helium", "Methane"]),
    ("What is the hardest natural mineral on the Mohs scale?", "Diamond", ["Quartz", "Topaz", "Corundum"]),
]

PASSAGE_LIGHTHOUSE = (
    "The Marrow Point lighthouse was built in 1889 after three coastal ships were lost on the same reef "
    "in a single winter. Its lamp originally burned whale oil, but the keepers switched to kerosene in 1911, "
    "which made the light brighter and cut the need for deliveries by half. In 1962 the station was "
    "automated, and the last keeper, Helen Voss, left the island in the spring of that year. The tower is "
    "still standing, though the lamp now runs on electricity and is checked by radio from the mainland."
)
PASSAGE_LIBRARY = (
    "The Eastfield Library lends books for three weeks. Members may renew an item once, but only if no one "
    "has reserved it. Late returns cost ten cents per day per book, and fines above five dollars block further "
    "borrowing until they are paid. Rare manuscripts never leave the reading room. Children under twelve may "
    "borrow up to four books at a time, while adults may borrow up to twelve."
)
PASSAGE_BRIDGE = (
    "Inspectors examined the Calder Bridge in March. They found cracks in two of the six support piers, both "
    "near the waterline, and measured the cracks at 2 and 4 millimeters wide. The engineers decided that cracks "
    "under 3 millimeters could be monitored, while wider cracks required the pier to be repaired before the "
    "bridge could carry heavy trucks. The repair of the wider crack began in June and finished in August."
)
READING = [
    (PASSAGE_LIGHTHOUSE, "When was the Marrow Point lighthouse built?", "1889", ["1871", "1911", "1962"]),
    (PASSAGE_LIGHTHOUSE, "Why was the lighthouse built?",
     "Three ships were lost on the same reef in one winter",
     ["The keepers wanted a brighter lamp", "Whale oil was no longer available", "The island needed a radio station"]),
    (PASSAGE_LIGHTHOUSE, "What fuel did the lamp originally burn?", "Whale oil",
     ["Kerosene", "Electricity", "Coal gas"]),
    (PASSAGE_LIGHTHOUSE, "According to the passage, what change reduced the need for deliveries by half?",
     "Switching to kerosene", ["Automating the station", "Building the tower", "Adding a radio"]),
    (PASSAGE_LIGHTHOUSE, "When did the last keeper leave the island?", "In the spring of 1962",
     ["In the winter of 1911", "In 1889", "In 1989"]),
    (PASSAGE_LIBRARY, "How long is the standard loan period?", "Three weeks",
     ["Two weeks", "One month", "Four weeks"]),
    (PASSAGE_LIBRARY, "When may a member renew an item?", "Once, when no one has reserved it",
     ["Any number of times", "Only after a fine is paid", "Whenever a reservation is made"]),
    (PASSAGE_LIBRARY, "What happens when fines exceed five dollars?",
     "Further borrowing is blocked until they are paid",
     ["The book is confiscated", "The loan period is doubled", "The fine is waived"]),
    (PASSAGE_LIBRARY, "How many books may a child under twelve borrow at once?", "Four",
     ["Twelve", "Two", "Six"]),
    (PASSAGE_LIBRARY, "Which items never leave the reading room?", "Rare manuscripts",
     ["Children's books", "Reserved items", "Overdue books"]),
    (PASSAGE_BRIDGE, "How many support piers had cracks?", "Two", ["Six", "Four", "One"]),
    (PASSAGE_BRIDGE, "Where were the cracks located?", "Near the waterline",
     ["At the top of the deck", "Under the road surface", "Near the bridge's midpoint"]),
    (PASSAGE_BRIDGE, "What crack width threshold required a repair before heavy trucks could pass?",
     "Wider than 3 millimeters", ["Wider than 2 millimeters", "Wider than 5 millimeters",
                                  "Wider than 10 millimeters"]),
    (PASSAGE_BRIDGE, "When did the repair of the wider crack begin?", "June", ["March", "August", "April"]),
    (PASSAGE_BRIDGE, "Which crack was repaired, and when did the repair end?",
     "The 4 millimeter crack; in August",
     ["The 2 millimeter crack; in June", "Both cracks; in March", "The 4 millimeter crack; in June"]),
]

# (id, prompt, verifier spec, passing reference, failing counterexample)
FORMAT = [
    ("Return a JSON object with the keys name and city for a fictional person. Output only the JSON.",
     {"type": "json_keys", "keys": ["city", "name"]}, '{"name": "Ira", "city": "Lisbon"}', '{"name": "Ira"}'),
    ("Return a JSON object with exactly the keys width, height and depth describing a box. Output only JSON.",
     {"type": "json_keys", "keys": ["depth", "height", "width"]}, '{"width": 2, "height": 3, "depth": 4}',
     '{"width": 2, "height": 3, "depth": 4, "color": "red"}'),
    ("Output a JSON object with keys title and year for any novel you like, and nothing else.",
     {"type": "json_keys", "keys": ["title", "year"]}, '{"title": "Dune", "year": 1965}',
     'Here it is: {"title": "Dune", "year": 1965}'),
    ("Return a JSON object with keys ids (a list of three integers) and total (their sum). Output only JSON.",
     {"type": "json_keys", "keys": ["ids", "total"]}, '{"ids": [1, 2, 3], "total": 6}', '{"ids": [1, 2, 3]}'),
    ("In at most 20 words, explain why the sky is blue.",
     {"type": "max_words", "n": 20},
     "Sunlight scatters off air molecules, and shorter blue wavelengths scatter more strongly than red.",
     " ".join(["word"] * 25)),
    ("Summarize the water cycle in twelve words or fewer.",
     {"type": "max_words", "n": 12},
     "Water evaporates, condenses into clouds, falls as rain, and returns to oceans.",
     "The water cycle is a continuous process that moves water through the planet in many stages."),
    ("Reply with a single word that names a primary color.",
     {"type": "max_words", "n": 1}, "Blue", "Blue green"),
    ("Write at least 40 words about the sea.",
     {"type": "min_words", "n": 40}, " ".join(["The sea rises and the sea falls."] * 8), "Sea waves roll in."),
    ("Start your reply with the word Answer: followed by your answer to 7 times 6.",
     {"type": "starts_with", "text": "Answer:"}, "Answer: 42", "42"),
    ("Write a two-sentence note that begins with Dear.",
     {"type": "starts_with", "text": "Dear"}, "Dear team, the report is ready. Please review it today.",
     "Hello team, the report is ready."),
    ("Write a short greeting and end it with the exact phrase Thank you.",
     {"type": "ends_with", "text": "Thank you."}, "Hello from the lab, we are ready to begin. Thank you.",
     "Hello from the lab. Thanks."),
    ("Write one sentence about rivers using only lowercase letters.",
     {"type": "all_lowercase"}, "rivers carry sediment from the hills to the sea.", "Rivers carry sediment."),
    ("Write a haiku in lowercase letters only.",
     {"type": "all_lowercase"}, "soft rain on the roof\nthe river keeps its own time\nmorning finds the path",
     "Soft rain on the roof"),
    ("Describe a bicycle in one sentence without using any commas.",
     {"type": "no_commas"}, "A bicycle has two wheels and a light frame.",
     "A bicycle has two wheels, a chain, and a frame."),
    ("List exactly three reasons to drink water, each as a bullet starting with '- '. Use no other lines.",
     {"type": "bullet_count", "n": 3},
     "- It regulates body temperature.\n- It carries nutrients to cells.\n- It keeps joints lubricated.",
     "- It regulates body temperature.\n- It carries nutrients to cells."),
    ("List four planets as bullets beginning with '- '. Use no other lines.",
     {"type": "bullet_count", "n": 4}, "- Mercury\n- Venus\n- Earth\n- Mars",
     "Here are four:\n- Mercury\n- Venus\n- Earth\n- Mars"),
    ("Write a sentence that uses the words lantern, harbor and quiet.",
     {"type": "contains_all", "words": ["lantern", "harbor", "quiet"]},
     "The lantern glowed softly over the quiet harbor.", "The lantern glowed over the harbor."),
    ("Name three fruits in one sentence with no commas.",
     {"type": "no_commas"}, "Apples and pears and plums are common fruits.",
     "Apples, pears, and plums are common fruits."),
    ("List five prime numbers under 20 as bullets beginning with '- ' and nothing else.",
     {"type": "bullet_count", "n": 5}, "- 2\n- 3\n- 5\n- 7\n- 11", "- 2\n- 3\n- 5\n- 7"),
    ("Give installation instructions for a kettle in three steps, starting with the text Step 1.",
     {"type": "starts_with", "text": "Step 1"}, "Step 1: fill the kettle.\nStep 2: plug it in.\nStep 3: press start.",
     "First, fill the kettle."),
]
FORMAT_SPECS = {"json_keys", "max_words", "min_words", "starts_with", "ends_with", "all_lowercase",
                "no_commas", "bullet_count", "contains_all"}


def verify(response: str, spec: dict) -> bool:
    """Return True when a response satisfies one programmatic format spec. Unknown types raise."""
    kind = spec["type"]
    if kind == "json_keys":
        try:
            parsed = json.loads(response)
        except ValueError:
            return False
        return isinstance(parsed, dict) and sorted(parsed) == sorted(spec["keys"])
    if kind == "max_words":
        return 0 < len(response.split()) <= spec["n"]
    if kind == "min_words":
        return len(response.split()) >= spec["n"]
    if kind == "starts_with":
        return response.startswith(spec["text"])
    if kind == "ends_with":
        return response.rstrip().endswith(spec["text"])
    if kind == "all_lowercase":
        return bool(response.strip()) and response == response.lower()
    if kind == "no_commas":
        return bool(response.strip()) and "," not in response
    if kind == "bullet_count":
        lines = [line for line in response.splitlines() if line.strip()]
        bullets = [line for line in lines if line.startswith("- ")]
        return len(bullets) == spec["n"] and len(bullets) == len(lines)
    if kind == "contains_all":
        lowered = response.lower()
        return all(word.lower() in lowered for word in spec["words"])
    raise ValueError(f"Unknown verifier type: {kind}")


def run_reference(code: str):
    """Execute a coding reference snippet in a fresh isolated interpreter and return its `result`."""
    harness = code + "\nimport json as _json\nprint(_json.dumps(result))\n"
    proc = subprocess.run([sys.executable, "-I", "-c", harness], capture_output=True, text=True,
                          timeout=30, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Reference snippet failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


def build_coding():
    cases = []
    for index, (code, question, expected) in enumerate(CODING, 1):
        executed = run_reference(code)
        if canonical(executed) != canonical(expected):
            raise ValueError(f"Coding case {index} expected {canonical(expected)} but executed {canonical(executed)}")
        cases.append({"case_id": f"wsp-coding-{index:02d}", "code": code, "question": question,
                      "expected": expected})
    return {"purpose": CODING_PURPOSE, "cases": cases}


def place(correct: str, distractors: list[str], index: int):
    """Put the correct option at a rotating position so answer letters are balanced."""
    options = list(distractors)
    options.insert(index, correct)
    return options, LETTERS[index]


def build_general():
    items = []
    entries = []
    for question, expr in ARITHMETIC:
        value = eval(expr, {}, {})  # trusted constant expressions written in this file
        if value != int(value):
            raise ValueError(f"Arithmetic answer is not an integer: {question}")
        value = int(value)
        entries.append(("arithmetic", question, str(value), [str(value + 5), str(value + 10), str(value - 3)]))
    for question, correct, distractors in LOGIC:
        entries.append(("logic", question, correct, distractors))
    for question, correct, distractors in SCIENCE:
        entries.append(("science", question, correct, distractors))
    for passage, question, correct, distractors in READING:
        entries.append(("reading_comprehension", f"Passage: {passage}\n\nQuestion: {question}", correct, distractors))
    for index, (topic, question, correct, distractors) in enumerate(entries):
        if len(set(distractors + [correct])) != 4:
            raise ValueError(f"Options are not distinct: {question}")
        options, letter = place(correct, distractors, index % 4)
        items.append({"id": f"wsp-general-{index + 1:02d}", "question": question, "options": options,
                      "answer": letter, "topic": topic})
    return {"purpose": GENERAL_PURPOSE, "cases": items}


def build_format():
    cases = []
    for index, (prompt, spec, reference, counterexample) in enumerate(FORMAT, 1):
        if spec["type"] not in FORMAT_SPECS:
            raise ValueError(f"Unknown verifier type: {spec['type']}")
        if not verify(reference, spec):
            raise ValueError(f"Reference fails its own verifier: format item {index}")
        if verify(counterexample, spec):
            raise ValueError(f"Counterexample passes its verifier: format item {index}")
        cases.append({"id": f"wsp-format-{index:02d}", "prompt": prompt, "verifier": spec})
    return {"purpose": FORMAT_PURPOSE, "cases": cases}


def dump(document) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(root: Path) -> dict:
    coding, general, format_document = build_coding(), build_general(), build_format()
    outputs = {
        "datasets/wsp-coding.json": dump(coding),
        "datasets/wsp-general.json": dump(general),
        "datasets/wsp-format.json": dump(format_document),
    }
    letters = {letter: sum(case["answer"] == letter for case in general["cases"]) for letter in LETTERS}
    summary = {
        "schema_version": 1,
        "builder": "scripts/build_wsp_regressions.py",
        "files": {
            "datasets/wsp-coding.json": {
                "kind": "coding", "cases": len(coding["cases"]),
                "verification": f"{len(coding['cases'])}/{len(coding['cases'])} expected answers matched "
                                "when the reference snippet was executed in a subprocess",
            },
            "datasets/wsp-general.json": {
                "kind": "multiple_choice", "cases": len(general["cases"]),
                "answer_letter_counts": letters,
                "verification": "arithmetic answers recomputed; four distinct options; one correct letter",
            },
            "datasets/wsp-format.json": {
                "kind": "instruction_following", "cases": len(format_document["cases"]),
                "verifier_types": sorted({case["verifier"]["type"] for case in format_document["cases"]}),
                "verification": "each passing reference satisfies its verifier; each counterexample fails",
            },
        },
    }
    for name, data in outputs.items():
        summary["files"][name]["sha256"] = sha256(data)
        summary["files"][name]["bytes"] = len(data)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    config_path = root / "configs" / "wsp-regressions.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(dump(summary))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root to write under")
    args = parser.parse_args(argv)
    summary = build(args.root.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
