from extraction_validator.judge import get_judge_inputs
from fire import Fire # type: ignore

def main(
    ard_dir: str,
    processed_dir: str,
    output_dir: str,
):
    judge_inputs, output_path = get_judge_inputs(
        ard_dir=ard_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )

    for judge_input in judge_inputs:
        with open(
            output_path / f"{judge_input.file_name}.json", "w"
        ) as f:
            f.write(
            f"""Original Text:
{judge_input.original_text}
Extraction Output:
{judge_input.kg_output.model_dump_json(indent=2)}
            """)
if __name__ == "__main__":
    Fire(main)