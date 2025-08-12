# Story 2

## Intervention causal chain prompt development

Robust prompt for eliciting correct and comprehensive extraction of causal chain of concepts with all proposed interventions identified per paper (can start with Buehler 2024 and Tong 2024); focus on capturing specific, rather than general, concepts that lead to the proposed intervention(s) in a given paper; for each intervention focus on level of detail to extract about 5-20 concepts that are each needed to follow how the final intervention is proposed (will be optimized on first data, may orient at level of masters in ML/AI to remain compact but not requiring expert understanding)

## Structure

- **entity-extraction.py** jupyter notebook with framework for testing and evaluating multiple prompt strategies
- **prompts.py** all text prompts as multi-line strings
- **plot_json_graphviz.py** 

- **all_papers.json** dynamic list of all papers in the /input folder (subset for testing)

- **/input** subset of papers for testing
- **/traces** freeform 'reasoning traces' from prompt tests
- **/json** structured json of nodes from prompt tests