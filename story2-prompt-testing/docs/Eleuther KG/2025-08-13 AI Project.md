---
tags: 
---
- [Google Search - llm evaluation metrics](https://www.google.com/search?client=safari&rls=en&q=metrics+to+evaluate+llm+prompts&ie=UTF-8&oe=UTF-8)
	- <https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5>
	- few-shot prompting example of LLM-driven evaluation for NER tasks.
	- <https://klu.ai/glossary/llm-evaluation>
- How atomic do we need to go in the Nodes..
- Once we get Story2 (mostly) delivered, I'll focus on downstream stories incl systemtic study on the prompt method choice. graphs, metrics etc., bc this will be published!
- Talk to Story3 team about any handling requirements
- Story X visualizations
- Story 11 manual validation
- deprioritize the other Issues
- what is ARD
- check the other staging branches
- terminology: freeform > rationale or reasoning trace
- pull out the hard-coded items from json such as paper title and id
	- do we need to include the aliases
	- do we need to strictly define the source, target, chain id? (yes, its inconsistent) do we need to add a paper-id (yes?)
	- what are the requirements (if any) for the various ids? might understand by plugging in the FlakorDB.

[[2025-08-14]]

- Approach
	- Design prompts
		- First qualitative test of reasoning trace -- 2307.16513v2 "Deception Abilities Emerged in Large Language Models" paper, don't ask for json, compare to list of instructions the response should have addressed. Run same prompt multiple X and evaluate stability. We're wanting an output similar to the Gemini example.
		- Include Dmitri's category taxonomy as an alternative?
	- Design schemas
		- Add the 'aliases' step in the json pass
		- Remove paper DOI, other metadata (append manually)
		- define format for node ids etc. (Do we need a separate alphanum ID or is the title sufficient)
	- Design tests and evals
		- Single pass, separate response and json as struct output
			- Closed this option, replicate later for the report
			- Double-check this is still failing? Worked for small tests.
		- Single pass, everything in json
			- Closed this option, replicate later for the report
		- Single pass, everything in text response
			- Do we get reliable json across multiple papers and trials? If not, close
		- Double pass, paper and reasoning in context
			- Evaluate quality and token cost
		- Double pass, reasoning in context
			- Evaluate quality and token cost
	- Evaluation criteria
		- Tokens
		- does json validate (y/n)
		- do json nodes track back to specific reasoning traces? (and vice-versa) (y/n/%)
		- quality/consistency of 'maturity' and 'confidence' scores and categories
		- quality of reasoning traces (qualitative)
		- graphs of nodes
	- Response Evaluation Criteria
		- Concept and Intervention nodes clearly defined.
			- Specific, clear, standalone, compact representations
			- If complex Interventions, represented as multiple steps with 'implemented_by' edges
			- Edge Confidence and Intervention Maturity scores always present, make sense.
		- Logical Chains and edges
			- Chain Integrity - flows from problem through supporting concepts to actionable intervention
			- (Relationship labels) - does the model go beyond defined examples?
			- (Maturity and Confidence rubrics, Step 6) We may be able to include more explicit and quantified rubrics here, please feel free to use LLMs to develop for both maturity and confidence. This would help get more self-consistent assessments.
		- Inference Strategy
			- Adopts and describes appropriate inference strategy for non-AI papers as instructed
			- Label is not so important (Martin)
		- Overall Format
			- Output is consistent between models and between runs.
				- Looks solved with a few-shot on the output format.

# Evals

- ROUGE, METEOR, CIder etc (see chats).. possible as part of the systemic study but in the interest of speed perhaps just human judgment on the rationale trace, structured evaluation of the json, and a token count.

# Reasoning Trace Format

Provide as a few-shot/instructions

- Summary
	- Robust summary of the findings of the paper.
	- Summary of limitations, uncertainties or identified gaps in the paper.
	- Describe Inference Strategy used and rationale.
- Logical Chains: logical chains extracted from the paper, structured as concept and intervention nodes connected by relationship edges. Provide as text, not a table.
	- Logical Chain 1
		- Robust summary of the Logical Chain and rationale.
		- Iterate the following structure for each Node-to-Node relationship in the Logical Chain:
			- Source Node: One-sentence unique node description. (label Concept or Intervention)
			- Edge Type: Edge relationship label.
			- Target Node: One-sentence unique node description. (label Concept or Intervention)
			- Edge Confidence: Edge confidence label.
			- Edge Confidence Rationale: Detailed edge confidence rationale.
			- Intervention Maturity: Intervention maturity label (if Intervention.)
			- Intervention Maturity Rationale: detailed intervention maturity rationale (if Intervention.)
	- Iterate over all Logical Chains in paper.
