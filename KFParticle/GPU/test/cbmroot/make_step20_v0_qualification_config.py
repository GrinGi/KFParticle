#!/usr/bin/env python3
"""Derive a reviewed Step 20.3 V0-only YAML from a complete MainConfig.yaml."""

import re
import sys
from pathlib import Path


def indentation(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def significant(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.yaml OUTPUT.yaml", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)

    output = []
    in_kfp = False
    in_selector = False
    in_decays = False
    in_finder_cuts = False
    decay_indent = -1
    finder_indent = -1
    item = []
    item_pdg = None
    found_decays = False
    found_finder_cuts = False
    retained_pdgs = []
    allowed_pdgs = {310, 3122, -3122}
    source_has_anti_lambda = any(
        re.match(r"^\s*-\s*pdg:\s*-3122\b", line) for line in lines
    )

    def flush_item() -> None:
        nonlocal item, item_pdg
        if item and item_pdg in allowed_pdgs:
            output.extend(item)
            retained_pdgs.append(item_pdg)
            if item_pdg == 3122 and not source_has_anti_lambda:
                conjugate = list(item)
                conjugate[0] = re.sub(
                    r"(pdg:\s*)3122\b", r"\g<1>-3122", conjugate[0]
                )
                conjugate[0] = re.sub(r"#.*$", "# anti-Lambda", conjugate[0])
                output.extend(conjugate)
                retained_pdgs.append(-3122)
        item = []
        item_pdg = None

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        indent = indentation(line)

        if in_finder_cuts:
            if significant(line) and indent <= finder_indent:
                in_finder_cuts = False
                continue
            index += 1
            continue

        if in_decays:
            match = re.match(r"^\s*-\s*pdg:\s*(-?\d+)\b", line)
            if match:
                flush_item()
                item = [line]
                item_pdg = int(match.group(1))
                index += 1
                continue
            if item and (not significant(line) or indent > decay_indent):
                item.append(line)
                index += 1
                continue
            flush_item()
            in_decays = False
            continue

        if significant(line) and indent == 0:
            in_kfp = stripped == "kfp:"
            in_selector = False
        elif in_kfp and significant(line) and indent == 2:
            in_selector = stripped == "selector:"

        if in_selector and stripped == "decays:":
            found_decays = True
            in_decays = True
            decay_indent = indent
            output.append(line)
            index += 1
            continue
        if in_selector and stripped == "finderCuts:":
            found_finder_cuts = True
            in_finder_cuts = True
            finder_indent = indent
            index += 1
            continue

        output.append(line)
        index += 1

    flush_item()
    if not found_decays or not retained_pdgs:
        print("input does not contain a non-empty kfp.selector.decays V0 list", file=sys.stderr)
        return 2
    if not found_finder_cuts:
        print("input does not contain kfp.selector.finderCuts; refusing an unreviewed shape", file=sys.stderr)
        return 2
    if set(retained_pdgs) != allowed_pdgs:
        print("qualification configuration must contain K0S, Lambda, and anti-Lambda", file=sys.stderr)
        return 2
    if any(pdg not in allowed_pdgs for pdg in retained_pdgs):
        print("internal error: unsupported decay survived", file=sys.stderr)
        return 2

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(output), encoding="utf-8")
    print(
        "PASS step20-v0-qualification-config - retained PDGs "
        + ",".join(str(pdg) for pdg in retained_pdgs)
        + "; removed finderCuts"
    )
    print(f"  output     : {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
