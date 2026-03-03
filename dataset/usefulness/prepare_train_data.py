from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ID_SUFFIX_PATTERN = re.compile(r"-(u|nu)$", re.IGNORECASE)


def infer_label(record_id: str | None, file_name: str) -> int | None:
	if record_id:
		match = ID_SUFFIX_PATTERN.search(record_id.strip())
		if match:
			return 1 if match.group(1).lower() == "u" else 0

	lowered = file_name.lower()
	if "non-useful" in lowered or "non_useful" in lowered:
		return 0
	if "useful" in lowered:
		return 1
	return None


def parse_line(raw_line: str) -> tuple[str | None, str | None]:
	line = raw_line.strip()
	if not line:
		return None, None

	# Expected format: <record_id>\t<message>
	tab_parts = line.split("\t", maxsplit=1)
	if len(tab_parts) == 2:
		record_id, message = tab_parts[0].strip(), tab_parts[1].strip()
		return (record_id or None), (message or None)

	# Fallback format: <record_id> <message>
	space_parts = line.split(maxsplit=1)
	if len(space_parts) == 2 and ID_SUFFIX_PATTERN.search(space_parts[0]):
		record_id, message = space_parts[0].strip(), space_parts[1].strip()
		return (record_id or None), (message or None)

	return None, line


def load_labeled_messages(input_dir: str | Path) -> pd.DataFrame:
	base_dir = Path(input_dir)
	if not base_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {base_dir}")

	rows: list[dict[str, object]] = []

	for txt_file in sorted(base_dir.rglob("*.txt")):
		for line_number, raw_line in enumerate(
			txt_file.read_text(encoding="utf-8", errors="ignore").splitlines(),
			start=1,
		):
			record_id, message = parse_line(raw_line)
			if not message:
				continue

			label = infer_label(record_id, txt_file.name)
			if label is None:
				continue

			rows.append(
				{
					"message": message.strip(),
					"is_useful": int(label),
					"source_file": txt_file.name,
					"source_line": line_number,
				}
			)

	if not rows:
		return pd.DataFrame(columns=["message", "is_useful", "source_file", "source_line"])

	frame = pd.DataFrame(rows, columns=["message", "is_useful", "source_file", "source_line"])
	frame = frame[frame["message"] != ""].reset_index(drop=True)
	return frame


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Load labeled review-comment corpora and export one CSV with message, is_useful, source_file, and source_line columns."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path(__file__).parent / "rev_helper" / "comparative-study",
		help="Directory containing .txt corpus files",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path(__file__).parent / "messages_usefulness.csv",
		help="Output CSV path",
	)
	args = parser.parse_args()

	dataset = load_labeled_messages(args.input_dir)
	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	dataset.to_csv(args.output_csv, index=False)

	useful_count = int((dataset["is_useful"] == 1).sum()) if not dataset.empty else 0
	non_useful_count = int((dataset["is_useful"] == 0).sum()) if not dataset.empty else 0
	print(f"Saved {len(dataset)} rows to {args.output_csv}")
	print(f"is_useful=1: {useful_count}")
	print(f"is_useful=0: {non_useful_count}")


if __name__ == "__main__":
	main()
