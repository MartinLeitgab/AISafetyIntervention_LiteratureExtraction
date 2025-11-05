This is the extraction validator aka the JUDGE LLM.

It is used to validate the output of the information extraction LLMs.

Usage:

I have provided some test arxiv and a small subset of the ard for testing. 


```bash
# Run from the root directory
uv run python3 ./intervention_graph_creation/src/local_graph_extraction/llm_judge/judge.py \
    --processed_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_processed \
    --ard_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_ard \
    --output_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_output \
    --how_many_batches_in_flight_at_once=5 \
    --batch_size=50
```
--------
If you have an errors processing a batch, it will save the errors in a file called error.json. You can use this error file to re-process only the failed items like so:

```bash
# This will copy the directories of the files that had processing
# error last time into the new folder
uv run python3 ./intervention_graph_creation/src/local_graph_extraction/llm_judge/create_directory_for_rety.py \
    --last_output_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_output \
    --last_processed_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_processed \
    --new_folder_for_retry=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_retry
```

Then you can run again using the new retry folder as input for the processed dir, you can leave the output dir the same so that it appends to the previous output, and the ard dir the same.
```bash
# Run from the root directory
uv run python3 ./intervention_graph_creation/src/local_graph_extraction/llm_judge/judge.py \
    --processed_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_retry \
    --ard_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_ard \
    --output_dir=./intervention_graph_creation/src/local_graph_extraction/llm_judge/test_output \
    --how_many_batches_in_flight_at_once=5 \
    --batch_size=50
```