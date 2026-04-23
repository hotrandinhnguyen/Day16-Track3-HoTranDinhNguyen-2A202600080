from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    react_jsonl = out_path / "react_runs.jsonl"
    reflexion_jsonl = out_path / "reflexion_runs.jsonl"

    react_records = []
    with react_jsonl.open("w", encoding="utf-8") as f:
        for i, example in enumerate(examples, 1):
            print(f"[ReAct] {i}/{len(examples)}: {example.question[:60]}...")
            record = react.run(example)
            react_records.append(record)
            f.write(record.model_dump_json() + "\n")
            f.flush()

    reflexion_records = []
    with reflexion_jsonl.open("w", encoding="utf-8") as f:
        for i, example in enumerate(examples, 1):
            print(f"[Reflexion] {i}/{len(examples)}: {example.question[:60]}...")
            record = reflexion.run(example)
            reflexion_records.append(record)
            f.write(record.model_dump_json() + "\n")
            f.flush()
    all_records = react_records + reflexion_records
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
