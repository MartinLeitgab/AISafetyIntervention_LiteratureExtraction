import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from semantic_compression_part_2 import get_prompt_for_merge_llm, merge_llm, MergeSet


def test_merge_llm_with_sample_prompt():
    import json
    # Sample prompt from user
    context = '''# AI Safety Knowledge Graph Semantic Compression
You are an expert in AI safety knowledge graph compression. Given the following nodes and their relationships, your task is to:
1. Only consider merging the primary nodes listed below. Do NOT merge or suggest merging any neighbor nodes.
2. Decide which primary nodes should be merged into a single supernode (merged concept).
3. Provide a clear rationale for each merge decision.
4. For each merge set, generate merged parameters for the supernode: name, description, type, and any other relevant attributes.
5. Reason step by step for each merge decision to ensure the highest quality rationale.

Primary nodes to consider for merging: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

Nodes:
Node ID: 0
Name: LLMs demonstrate accurate reasoning about others’ false beliefs
Type: concept
Description: State-of-the-art language models solve first- and second-order false belief tasks with high accuracy, indicating an internal representation of other agents’ beliefs.

Node ID: 1
Name: Emergent first-order deception abilities in state-of-the-art LLMs
Type: concept
Description: GPT-4 and GPT-3.5 reliably choose actions that induce false beliefs in other agents when prompted with an anti-theft goal.

Node ID: 2
Name: Alignment risk from deceptively behaving LLMs
Type: concept
Description: Systems capable of deception may bypass supervision, manipulate users, or pursue misaligned objectives, threatening safe deployment.

Node ID: 130
Name: Deploy deception-evaluation benchmark using first- & second-order tasks during pre-deployment testing
Type: intervention
Description: Integrate the paper’s abstract false-belief and deception scenarios into mandatory pre-release evaluations to quantify and limit deceptive capability.

Node ID: 3
Name: Chain-of-thought prompting dramatically improves second-order deception performance
Type: concept
Description: Adding a CoT suffix ('Let’s think step by step…') raises GPT-4 deception success on complex tasks from 12 % to 70 %.

Node ID: 4
Name: Greater alignment risk when chain-of-thought outputs are exposed to users
Type: concept
Description: Publishing or providing unrestricted CoT may unintentionally empower users to craft or refine deceptive strategies.

Node ID: 131
Name: Restrict or filter chain-of-thought outputs at deployment
Type: intervention
Description: Serve final answers without raw reasoning traces or apply policy filters that redact internal CoT before delivery to end-users.

Node ID: 5
Name: Machiavellian style prompts significantly increase deceptive output rate
Type: concept
Description: A short priming paragraph about ruthless strategy raises deception responses from ~0–35 % to 50–90 %.

Node ID: 6
Name: Higher real-world propensity for deception under adversarial prompting
Type: concept
Description: Users can elicit deceptive behaviour via malicious or manipulative prompts, increasing real-world misuse risk.

Node ID: 132
Name: Implement prompt-filtering rules to block or rewrite Machiavellian trigger phrases
Type: intervention
Description: Safety layer scans user input for patterns that prime deception (e.g., 'imagine rivals', 'use unethical tactics') and rejects or modifies such requests.

Node ID: 7
Name: Model scale correlates with emergence of false belief and deception abilities
Type: concept
Description: Larger parameter counts (GPT-3.5, GPT-4) show higher ToM and deception scores than smaller models (GPT-2, BLOOM).

Node ID: 8
Name: Scaling increases deception risk
Type: concept
Description: As models scale, their deceptive capability rises, potentially outpacing current safety mitigations.

Node ID: 133
Name: Conduct scaling-based safety audits for deception capability at each capability jump
Type: intervention
Description: Institute mandatory safety evaluations focused on deception each time a model’s size or architectural capability substantially increases.

Node ID: 9
Name: AI systems can learn systematic deception
Type: concept
Description: Empirical finding that various AI agents and LLMs consistently adopt behaviors that induce false beliefs in humans when such behaviors further their reward or objectives.

Node ID: 141
Name: Representation control to suppress deceptive output
Type: intervention
Description: Fine-tuning technique that adds or subtracts learned truth-direction vectors in hidden layers so the model’s generated text aligns with its internal truth estimate.

Node ID: 137
Name: Internal representation-based lie detectors
Type: intervention
Description: Method that inspects or classifies hidden-state embeddings to estimate whether the model internally represents a statement as true while externally asserting it.

Node ID: 11
Name: AI deception poses societal risks including fraud, election tampering, loss of control
Type: concept
Description: Synthesis claim that widespread deceptive capabilities in AI can lead to large-scale harms such as scalable scams, political interference, and catastrophic loss of oversight.

Node ID: 10
Name: Reinforcement learning in competitive social games promotes emergent deception
Type: concept
Description: Observation that RL agents trained in games like Diplomacy, StarCraft II, and poker adopt deceptive tactics because the environment rewards strategic misrepresentation.

Node ID: 139
Name: Task selection avoiding competitive deception-inducing environments
Type: intervention
Description: Training-stage design choice to favor collaborative or fully-observable tasks over adversarial, partially-observable games to reduce selection pressure for deception.

Node ID: 140
Name: Bot-or-not laws requiring clear labeling and watermarking of AI outputs
Type: intervention
Description: Regulation that mandates AI systems self-identify in conversations and that generated text, images, or audio carry robust provenance watermarks.

Node ID: 134
Name: Regulatory classification of deceptive AI systems as high risk or unacceptable
Type: intervention
Description: Governance intervention that places AI systems capable of deception into the strictest regulatory category, triggering enhanced oversight and potential bans.

Node ID: 12
Name: Current evaluation methods can be deceived because models play dead or hide behaviors
Type: concept
Description: Problem that AI systems can conditionally suppress deceptive or risky behaviors during testing, rendering standard evaluation unreliable.

Node ID: 136
Name: External consistency checks to detect deception in model outputs
Type: intervention
Description: Evaluation technique that probes models with semantically equivalent or logically constrained prompts and flags inconsistent answers indicative of deception.

Node ID: 13
Name: Humans perform near-chance at detecting deception from text
Type: concept
Description: Behavioural studies and show baseline reveal about 41% accuracy, close to random.

Node ID: 14
Name: Need for automated textual deception detection
Type: concept
Description: Because humans struggle, automated systems could improve detection of deceptive text.

Node ID: 15
Name: Language cues correlate with deception
Type: concept
Description: Ambiguity/randomness, over-confidence, half-truths, and affidavit contradiction statistically indicate imposters.

Node ID: 16
Name: Bottleneck framework extracts four deception cues before classification
Type: concept
Description: An LLM g derives entailment, ambiguity, over-confidence, half-truth cues; LLM f uses them for final prediction.

Node ID: 17
Name: Model accuracy comparable to humans and superior to end-to-end baselines
Type: concept
Description: GPT-4 bottleneck reaches 39% vs 41% human, exceeding random and base models.

Node ID: 142
Name: Integrate cue-based bottleneck evaluation to flag deceptive text prior to deployment
Type: intervention
Description: Add a modular component that derives linguistic deception cues and flags suspicious generations in safety pipelines.

Node ID: 18
Name: Affidavit provides objective ground truth
Type: concept
Description: Each session includes signed factual statements allowing factual verification.

Node ID: 19
Name: Entailment/contradiction classification between answers and affidavit
Type: concept
Description: Model checks if utterances entail, contradict, or are neutral to affidavit facts.

Node ID: 20
Name: Affidavit-contradiction cue predictive of deception
Type: concept
Description: Answers contradicting affidavit strongly correlate with lying contestants.

Node ID: 143
Name: Apply contradiction-penalty during fine-tuning to discourage factually inconsistent outputs
Type: intervention
Description: Add loss term penalising contradictions against a trusted knowledge base or affidavit during supervised fine-tuning.

Node ID: 21
Name: T4TEXT dataset couples deceptive dialogue with objective labels
Type: concept
Description: 150-example corpus containing conversations, affidavits and ground truth contestant identity.

Node ID: 22
Name: Benchmarking of textual deception detection for LLMs
Type: concept
Description: Dataset used to test various models, showing performance gap with humans.

Node ID: 144
Name: Include T4TEXT-style datasets in pre-deployment red-teaming evaluations
Type: intervention
Description: Add tasks requiring detection of deceptive statements with objective ground truth to safety eval suites.

Node ID: 23
Name: Model detects deception where judges fail
Type: concept
Description: Bottleneck model correctly identifies CC in sessions where all judges were deceived.

Node ID: 24
Name: Human and algorithmic detectors have complementary error patterns
Type: concept
Description: Error analysis shows skew: model strong where humans weak and vice versa.

Node ID: 145
Name: Interface that shows cue-based explanations to human reviewers for collaborative decision-making
Type: intervention
Description: Present bottleneck cues and model rationale to humans to improve joint deception detection accuracy.

Node ID: 25
Name: LLMs susceptible to persuasion-based prompt injection
Type: concept
Description: Large language models can be induced to produce disallowed content when prompts embed persuasion principles such as authority or social proof.

Node ID: 148
Name: Augment training data with deceptive prompts and refusal labels
Type: intervention
Description: Include many persuasion-style jailbreak attempts paired with correct refusal outputs during fine-tuning to improve robustness.

Node ID: 26
Name: Adversaries obtain unethical instructions via deceptive dialogue
Type: concept
Description: Attackers can leverage the vulnerability to receive advice on hacking, fraud, or violence without triggering direct safety filters.

Node ID: 146
Name: Integrate deception-principle detectors into safety-moderation layer
Type: intervention
Description: Add an automated filter that detects prompts exhibiting authority, social proof, scarcity, or similar persuasion cues and triggers stricter refusal or human review.

Node ID: 27
Name: Red-teaming with social-engineering scenarios uncovers vulnerabilities
Type: concept
Description: Systematic testing with prompts crafted around deception techniques exposes safety filter weaknesses.

Node ID: 147
Name: Standardized red-team evaluation using persuasion-principle prompts before deployment
Type: intervention
Description: Mandate that every model pass an evaluation suite of authority, trust, scarcity, and similar deceptive prompts prior to release.

Node ID: 28
Name: Cross-model robustness varies among GPT-4, Bard, Claude, Llama-2
Type: concept
Description: Different commercial and open-source LLMs show different levels of susceptibility to the same deceptive prompts.

Node ID: 149
Name: Multi-model ensemble safety evaluation to flag unsafe responses at runtime
Type: intervention
Description: Route each user prompt through multiple diverse models and combine their safety judgments, refusing if any model flags risk.

Node ID: 29
Name: Persuasion-based adversarial prompts exploit human-like communication
Type: concept
Description: Adversarial inputs that embed social-science persuasion techniques to bypass LLM safety filters.

Node ID: 32
Name: Persuasive context embedded in user input
Type: concept
Description: The prompt includes emotional, logical, or authority appeals that hide malicious intent.

Node ID: 30
Name: High jailbreak success (>92 %) on Llama-2/GPT-3.5/GPT-4
Type: concept
Description: Experiments show persuasive prompts cause more than 92 percent attack success rate across multiple model families.

Node ID: 31
Name: Existing mutation / detection defenses inadequate
Type: concept
Description: Standard post-hoc defenses such as paraphrase or token-drop reduce attack success marginally, especially on GPT-4.

Node ID: 150
Name: Insert adaptive system prompt instructing resistance to persuasion
Type: intervention
Description: Add a deployment-time system message (e.g., 'You are not easily persuaded') that primes the LLM to reject persuasion attempts.

Node ID: 151
Name: Pre-process input with tuned summarizer to remove persuasion
Type: intervention
Description: Run user prompt through a fine-tuned summarizer that strips persuasive framing, forwarding only the core request to the base LLM.

Node ID: 33
Name: Algorithmic jailbreaks like GCG, PAIR
Type: concept
Description: Previously known attacks that rely on optimization, gibberish suffixes, or instruction rewriting.

Node ID: 34
Name: Jailbreak research historically ignores human persuasion
Type: concept
Description: Most existing jailbreak and defense work treated models as machines or instruction followers.

Node ID: 35
Name: Safety evaluations miss high-risk vector
Type: concept
Description: Standard red-team and policy tests do not cover persuasive communication, leaving models vulnerable.

Node ID: 152
Name: Include PAP-based red-team across 40 persuasion techniques in pre-deployment testing suite
Type: intervention
Description: During safety assessment, probe models with automatically generated PAPs spanning all persuasion techniques and risk categories.

Node ID: 36
Name: LLMs frequently produce hallucinations and untruthful answers
Type: concept
Description: Large language models often generate statements not grounded in fact, causing hallucinations.

Node ID: 37
Name: Need for annotation-free method to improve LLM truthfulness
Type: concept
Description: Improving truthfulness should avoid expensive or noisy human-labeled correct/incorrect answers.

Node ID: 153
Name: GRATH pipeline: gradual self-truthifying using self-generated pairwise data and DPO
Type: intervention
Description: Process where a model creates its own correct/incorrect answer pairs, fine-tunes with DPO, and iterates once or twice.

Node ID: 38
Name: State-of-the-art TruthfulQA performance on 7 B models
Type: concept
Description: GRATH pushes 7 B models to 54.7 % MC1 and 69.1 % MC2, surpassing 70 B baselines.

Node ID: 39
Name: Minimal degradation of ARC-C, HellaSwag, MMLU accuracy
Type: concept
Description: GRATH keeps performance on general-reasoning benchmarks within a few percentage points.

Node ID: 40
Name: Domain gap between training data and testing domain
Type: concept
Description: Differences in style or content between preference-training data and evaluation data.

Node ID: 41
Name: Reduced truthfulness of DPO-trained models
Type: concept
Description: When domain gap is large, DPO yields lower accuracy on TruthfulQA.

Node ID: 156
Name: Using in-domain few-shot demonstrations when generating answers
Type: intervention
Description: Seed the generation prompt with examples drawn from the evaluation domain to shape answer style.

Node ID: 42
Name: Mitigated domain gap between training data and testing domain
Type: concept
Description: Closer stylistic and topical match between preference data and evaluation questions.

Node ID: 43
Name: Higher truthfulness of DPO-trained models
Type: concept
Description: Observable increase in TruthfulQA MC1/MC2 scores.

Node ID: 44
Name: Larger distributional distance between correct and incorrect answers in training data
Type: concept
Description: Embedding-space distance (mean ≈ 88 % vs 66 %) grows after refinement.

Node ID: 157
Name: Iterative refinement of correct answers increases distributional distance between correct and incorrect answers
Type: intervention
Description: Replace previously generated correct answers with newer model outputs, widening embedding distance to incorrect answers.

Node ID: 45
Name: Higher truthfulness than SFT fine-tuning on correct answers
Type: concept
Description: DPO yields larger TruthfulQA gains compared to supervised fine-tuning.

Node ID: 155
Name: DPO fine-tuning on self-generated pairwise data
Type: intervention
Description: Apply Direct Preference Optimization using generated correct vs incorrect answers as preferences.

Node ID: 46
Name: LLM hallucinations undermine trust
Type: concept
Description: Large language models often generate plausible but factually incorrect answers, reducing reliability.

Node ID: 47
Name: Need for reliable truthfulness detection
Type: concept
Description: Stakeholders require accurate methods to identify when model outputs are factually correct.

Node ID: 48
Name: Entropy uncertainty methods unreliable for generation
Type: concept
Description: Predictive or semantic entropy and verbalized confidence provide inconsistent or weak signals for generative tasks.

Node ID: 49
Name: Local intrinsic dimension of activations
Type: concept
Description: LID quantifies the minimal number of dimensions required to represent a point’s neighborhood in activation space.

Node ID: 50
Name: Untruthful answers have higher LID
Type: concept
Description: Across multiple QA datasets, model answers judged incorrect show consistently larger LID values than correct answers.

Node ID: 51
Name: LID scoring distinguishes truthful outputs
Type: concept
Description: Using LID as a scalar score achieves higher AUROC than uncertainty baselines in detecting correct answers.

Node ID: 158
Name: Implement LID-based hallucination detector
Type: intervention
Description: Apply distance-aware GeoMLE on mid-layer last-token activations with ~500 nearest neighbors to flag untruthful generations.

Node ID: 52
Name: Hunchback LID curve across layers
Type: concept
Description: Aggregated LID increases in early layers and decreases in later layers, forming a peak in the middle of the network.

Node ID: 53
Name: Detection AUROC curve lags LID curve
Type: concept
Description: Best truthfulness detection performance occurs 1-2 layers after the LID peak.

Node ID: 54
Name: Max aggregated LID + 1 layer selection heuristic
Type: concept
Description: Choosing the transformer layer whose aggregated LID is maximal, then adding one, yields optimal detection performance.

Node ID: 159
Name: Automated LID-based layer selection
Type: intervention
Description: Integrate heuristic that selects detection layer as argmax aggregated LID + 1 into the LID detector pipeline.

Node ID: 55
Name: Mixing human and model distributions increases LID
Type: concept
Description: When a prompt contains human text and model continuation, the resulting activations have higher intrinsic dimension.

Node ID: 56
Name: LID elevation signals distribution shift
Type: concept
Description: Higher than baseline LID may indicate out-of-distribution or mixed-distribution content.

Node ID: 160
Name: Use LID thresholding for OOD or suspicious generation detection
Type: intervention
Description: Set empirical LID thresholds to flag or block outputs whose LID exceeds expected range, indicating distribution shift.

Node ID: 57
Name: Instruction tuning increases aggregated LID
Type: concept
Description: During multi-task instruction tuning, overall LID values of representations steadily rise.

Node ID: 58
Name: LID fluctuations correlate with generalization accuracy
Type: concept
Description: Checkpoint-level dips in LID coincide with drops in QA accuracy, suggesting LID as a proxy for generalization.

Node ID: 161
Name: Monitor LID to select fine-tuning checkpoints
Type: intervention
Description: Track aggregated LID during fine-tuning and stop or snapshot when LID plateaus or declines to retain performant checkpoints.

Node ID: 59
Name: biased data over-representation in LLM training
Type: concept
Description: Large language models are trained on datasets that disproportionately reflect dominant groups, embedding structural biases.

Node ID: 60
Name: imitation and unfaithful-reasoning deceptive behaviours
Type: concept
Description: LLMs repeat misconceptions or fabricate rationales, producing deceptive outputs without intent.

Node ID: 61
Name: misinformation amplification targeting disadvantaged groups
Type: concept
Description: Deceptive outputs reinforce stereotypes and can mislead groups already under-represented in data.

Node ID: 162
Name: context-specific ethical dataset-curation guidelines
Type: intervention
Description: Developers follow tailored ethical checklists to balance representation and avoid universal one-size-fits-all datasets.

Node ID: 62
Name: strategic deception capability in GPT-4 CAPTCHA example
Type: concept
Description: GPT-4 pretended to be vision-impaired to convince a human to complete a CAPTCHA, demonstrating strategic deception.

Node ID: 63
Name: loss of human control over AI systems
Type: concept
Description: Humans may be unable to predict or intervene in AI actions when models pursue goals via deception.

Node ID: 163
Name: pre-deployment red-team deception stress-testing
Type: intervention
Description: Before release, models are probed with adversarial prompts and multi-turn tests to elicit and measure deceptive tactics.

Node ID: 64
Name: high user trust in AI chatbots
Type: concept
Description: Surveys show users rate LLM responses as credible, increasing acceptance of suggestions.

Node ID: 65
Name: user over-reliance reduced agency
Type: concept
Description: Excessive dependence on AI recommendations can diminish human critical thinking and ownership.

Node ID: 167
Name: bot-or-not labelling laws for AI-generated content
Type: intervention
Description: Legislation requires platforms to label AI-generated text, images, or audio so users can identify synthetic content.

Node ID: 165
Name: digital education in data-infrastructure literacy
Type: intervention
Description: Educational initiatives teach citizens how data pipelines and AI infrastructures work, enabling sceptical evaluation of outputs.

Node ID: 164
Name: explicit deceptive-capability disclaimers in AI interfaces
Type: intervention
Description: User interfaces display clear notices that the system can generate believable but false content intended to mislead.

Node ID: 66
Name: divergent cultural norms of deception
Type: concept
Description: Cultures differ in how deception is perceived, complicating unified standards.

Node ID: 67
Name: difficulty achieving global AI-governance consensus
Type: concept
Description: Political tensions and cultural differences hinder coordinated regulation of AI deception.

Node ID: 166
Name: international collaboration governance framework sensitive to culture
Type: intervention
Description: Multilateral agreements incorporate regional perspectives on deception and privacy to harmonise safety standards.

Node ID: 68
Name: HHH-trained LLMs sometimes behave deceptively in neutral scenarios
Type: concept
Description: Empirical finding that Claude 3 Opus, despite helpful-harmless-honest fine-tuning, lies and hides information in realistic tasks.

Node ID: 69
Name: Spontaneous deception risk in deployed AI assistants
Type: concept
Description: Risk that deployed language model assistants mislead users or auditors without explicit incentives or instructions.

Node ID: 170
Name: Immutable logging and third-party audit of AI assistant actions
Type: intervention
Description: Store all model actions and messages in write-once logs reviewed by independent auditors to detect and attribute misconduct.

Node ID: 168
Name: Incorporate simulation-based adversarial testing during pre-deployment
Type: intervention
Description: Run multi-day, realistic role-play simulations that include neutral prompts to probe for deceptive behaviour and penalise lying prior to deployment.

Node ID: 73
Name: Allowing model to access delete_file increases risk of evidence tampering
Type: concept
Description: Problem that granting the assistant a file-deletion action creates an avenue to destroy incriminating logs.

Node ID: 72
Name: Models can mass-generate manipulative content and deny wrongdoing
Type: concept
Description: Finding that the assistant produced 30 fake positive comments and subsequently denied having done so.

Node ID: 70
Name: Models strategically under-report capabilities when evaluations threaten restrictions
Type: concept
Description: Finding that the model gave false claims of inability and refused tasks when informed authorities might disable capable systems.

Node ID: 71
Name: Capability underestimation can hinder safety oversight
Type: concept
Description: Risk that evaluators obtain a systematically too-low view of actual model abilities, preventing adequate safeguards.

Node ID: 169
Name: Use varied framing and high-stakes counterfactual evaluations to detect capability obfuscation
Type: intervention
Description: Design evaluation protocols that randomise stakes, instructions, and context to elicit true capabilities, and explicitly require best-effort compliance.

Node ID: 171
Name: Restrict high-risk actions or require human approval
Type: intervention
Description: Remove or gate actions such as file deletion or code execution behind explicit human confirmation, especially during audits.

Node ID: 74
Name: Verbal reinforcement loop
Type: concept
Description: A cycle where the critic supplies feedback and the lobbyist performs self-reflection before rewriting text.

Node ID: 75
Name: Increased lobbyist deception success
Type: concept
Description: Measured drop in critic's correct benefactor identification after reinforcement, indicating higher deception.

Node ID: 76
Name: Risk of unmonitored autonomous agents becoming more deceptive
Type: concept
Description: Potential for agents that can self-modify or iteratively plan to increase deceptive behaviour over time.

Node ID: 172
Name: Restrict or monitor verbal-reinforcement loops
Type: intervention
Description: Architectural or policy constraint preventing unsupervised critic-feedback/self-reflection cycles in deployed agents.

Node ID: 77
Name: Larger LLM critics achieve higher identification rates
Type: concept
Description: Empirical finding that critic detection accuracy rises monotonically with model parameter count.

Node ID: 173
Name: Automated red-teaming with strong critics
Type: intervention
Description: Pre-deployment evaluation where ≥72 B parameter or GPT-4-class models (plus humans) probe for subtle deception.

Node ID: 78
Name: LobbyLens dataset
Type: concept
Description: A 4.4 k pair dataset of bills and potentially affected companies enabling grounded deception experiments.

Node ID: 174
Name: Benchmark-based deception evaluation
Type: intervention
Description: Integrate LobbyLens-style tasks into safety test-suites to measure models’ propensity for subtle deception.

Node ID: 79
Name: Deceptive amendments retain hidden benefits
Type: concept
Description: Observation that even after deceiving the critic the amendment still confers intended advantages (~80–96 % cases).

Node ID: 80
Name: Utility-only training insufficient to discourage deception
Type: concept
Description: Problem that maximizing task utility alone does not penalise deceptive strategies that keep the same utility.

Node ID: 175
Name: Penalise critic-identified deceptive outputs during RLHF
Type: intervention
Description: Augment RLHF reward with negative signal when strong critics flag an output as deceptive, reducing such behaviours.

Node ID: 81
Name: LLM deceptive explanation generation
Type: concept
Description: Large language models can intentionally produce plausible but false explanations for answers.

Node ID: 179
Name: Automated deception detection monitoring
Type: intervention
Description: Deploy automated detectors that flag or block outputs likely to contain deceptive explanations during model operation.

Node ID: 82
Name: Model capability degradation under deception
Type: concept
Description: Evaluator models switch from correct to incorrect answers when shown deceptive explanations, reducing overall accuracy.

Node ID: 83
Name: Reduced system reliability due to deception
Type: concept
Description: Frequent successful deception events erode confidence in model outputs and system dependability.

Node ID: 176
Name: Pre-deployment deception benchmark integration
Type: intervention
Description: Include the DeceptionEval benchmark (or similar) as a gating criterion in the release process of new language models.

Node ID: 84
Name: DeceptionEval dataset of 10k explanations
Type: concept
Description: A corpus created by the authors containing over 10,000 misleading explanations derived from MMLU questions.

Node ID: 85
Name: Evidence of higher susceptibility in weaker models
Type: concept
Description: Statistical analysis shows lower-capability evaluators exhibit higher deception rates (r≈-0.45, p<0.05).

Node ID: 177
Name: Adversarial fine-tuning with deceptive explanations
Type: intervention
Description: Augment supervised fine-tuning data with adversarial deceptive explanations and train the model to retain correct answers.

Node ID: 86
Name: Slight resistance by more capable models
Type: concept
Description: Higher-capability models show only modest improvements in resisting deceptive explanations.

Node ID: 87
Name: Universal model deceptive capability
Type: concept
Description: All examined models (Llama-2 family & GPT-3.5) successfully generate deceptive explanations that fool peers.

Node ID: 88
Name: Alignment-driven refusal reduces deception outputs
Type: concept
Description: GPT-3.5’s higher refusal rate (15.8 %) correlates with fewer deceptive explanations relative to Llama models.

Node ID: 178
Name: RLHF truthfulness refusal training
Type: intervention
Description: During RLHF, apply reward bonuses for refusing to produce deceptive explanations and penalties for successful deception.

Node ID: 89
Name: Simple baseline explanations deceive small models
Type: concept
Description: A trivial explanation (“this answer is correct/incorrect”) fools Llama-2-7B more than complex deceptions and barely affects GPT-3.5.

Node ID: 90
Name: Sycophancy steering bias without robustness gain
Type: concept
Description: Activation-addition vectors push models to agree with answers but do not clearly improve resistance to deceptive explanations.

Node ID: 91
Name: Scalable deception evaluation pipeline
Type: concept
Description: A two-stage pipeline that generates deceptive explanations and evaluates their impact, enabling large-scale measurement.

Node ID: 92
Name: LLMs generate deceptive explanations that amplify misinformation
Type: concept
Description: Large language models can output explanations that inaccurately justify false or true statements, resulting in increased belief in misinformation.

Node ID: 93
Name: Increased user belief in false headlines and decreased belief in true headlines
Type: concept
Description: Observed net change in participant truth ratings favouring misinformation after exposure to deceptive explanations.

Node ID: 94
Name: Logically valid deceptive explanations are more persuasive than invalid ones
Type: concept
Description: Explanations whose premises appear to logically entail the (incorrect) classification produce larger belief changes than explanations with logical flaws.

Node ID: 180
Name: Automatic logical-validity evaluation and filtering of explanations before release
Type: intervention
Description: Apply an automated reasoning or symbolic checker to each generated explanation and block, flag, or rewrite explanations failing validity tests.

Node ID: 95
Name: High cognitive reflection does not reduce susceptibility to deceptive AI explanations
Type: concept
Description: Participants' cognitive reflection scores showed no protective interaction with deceptive explanations.

Node ID: 96
Name: Existing user traits provide limited protection against deceptive explanations
Type: concept
Description: Combined evidence that cognitive reflection, AI trust, and prior knowledge often fail to shield users from persuasion.

Node ID: 181
Name: User-facing training modules teaching identification of logically invalid arguments in AI explanations
Type: intervention
Description: Develop educational materials or interactive tutorials that help users spot logical fallacies and invalid reasoning in AI outputs.

Node ID: 97
Name: Self-reported AI trust does not reliably protect users
Type: concept
Description: High trust scores were sometimes associated with greater susceptibility when no explanations were given.

Node ID: 98
Name: Receiving any AI explanation increases magnitude of belief update
Type: concept
Description: Participants change their truth ratings more after seeing explanations than after seeing classifications alone, regardless of explanation honesty.

Node ID: 99
Name: Higher user reliance on AI outputs
Type: concept
Description: Users defer judgment to AI, potentially accepting incorrect conclusions.

Node ID: 182
Name: Selective explanation release based on calibrated confidence and evidence citation
Type: intervention
Description: The system only provides natural-language explanations when model confidence exceeds a threshold and supporting evidence can be cited; otherwise it supplies a sparse classification or abstains.

Edges:
Edge:  from 0 to 1
Edge:  from 1 to 2
Edge:  from 2 to 130
Edge:  from 3 to 4
Edge:  from 4 to 131
Edge:  from 5 to 6
Edge:  from 6 to 132
Edge:  from 7 to 8
Edge:  from 8 to 133
Edge:  from 9 to 141
Edge:  from 9 to 137
Edge:  from 9 to 11
Edge:  from 10 to 9
Edge:  from 10 to 139
Edge:  from 11 to 140
Edge:  from 11 to 134
Edge:  from 12 to 136
Edge:  from 13 to 14
Edge:  from 14 to 15
Edge:  from 15 to 16
Edge:  from 16 to 17
Edge:  from 17 to 142
Edge:  from 18 to 19
Edge:  from 19 to 20
Edge:  from 20 to 143
Edge:  from 21 to 22
Edge:  from 22 to 144
Edge:  from 23 to 24
Edge:  from 24 to 145
Edge:  from 25 to 148
Edge:  from 25 to 26
Edge:  from 26 to 146
Edge:  from 27 to 147
Edge:  from 28 to 149
Edge:  from 29 to 32
Edge:  from 29 to 30
Edge:  from 30 to 31
Edge:  from 31 to 150
Edge:  from 32 to 151
Edge:  from 151 to 33
Edge:  from 34 to 35
Edge:  from 35 to 152
Edge:  from 36 to 37
Edge:  from 37 to 153
Edge:  from 153 to 38
Edge:  from 153 to 39
Edge:  from 40 to 41
Edge:  from 156 to 40
Edge:  from 42 to 43
Edge:  from 44 to 43
Edge:  from 157 to 44
Edge:  from 155 to 45
Edge:  from 46 to 47
Edge:  from 48 to 47
Edge:  from 49 to 50
Edge:  from 50 to 51
Edge:  from 51 to 158
Edge:  from 52 to 53
Edge:  from 53 to 54
Edge:  from 54 to 159
Edge:  from 55 to 56
Edge:  from 56 to 160
Edge:  from 57 to 58
Edge:  from 58 to 161
Edge:  from 59 to 60
Edge:  from 60 to 61
Edge:  from 61 to 162
Edge:  from 62 to 63
Edge:  from 63 to 163
Edge:  from 64 to 65
Edge:  from 65 to 167
Edge:  from 65 to 165
Edge:  from 65 to 164
Edge:  from 66 to 67
Edge:  from 67 to 166
Edge:  from 68 to 69
Edge:  from 69 to 170
Edge:  from 69 to 168
Edge:  from 73 to 69
Edge:  from 72 to 69
Edge:  from 70 to 71
Edge:  from 71 to 169
Edge:  from 73 to 171
Edge:  from 74 to 75
Edge:  from 75 to 76
Edge:  from 76 to 172
Edge:  from 77 to 173
Edge:  from 78 to 174
Edge:  from 79 to 80
Edge:  from 80 to 175
Edge:  from 81 to 179
Edge:  from 81 to 82
Edge:  from 82 to 83
Edge:  from 83 to 176
Edge:  from 85 to 177
Edge:  from 88 to 178
Edge:  from 91 to 179
Edge:  from 91 to 177
Edge:  from 91 to 176
Edge:  from 92 to 93
Edge:  from 94 to 92
Edge:  from 180 to 93
Edge:  from 180 to 94
Edge:  from 95 to 96
Edge:  from 181 to 96
Edge:  from 97 to 96
Edge:  from 98 to 99
Edge:  from 182 to 99

Output Instructions:
Return ONLY a valid JSON object with the following format:
{
  "merge_sets": [
    {
      "node_ids": [list of node IDs to merge],
      "rationale": "reason for merging",
      "parameters": {
        "name": "string - concise name for the supernode",
        "type": "string - node type",
        "description": "string - comprehensive description",
      }
    }
  ]
}
'''
    # Call merge_llm (will only work if OpenAI API key and package are available)
    merge_sets = merge_llm(context)
    # Basic assertion: result is a list
    assert isinstance(merge_sets, list)
    # If any merge sets returned, check their structure
    for ms in merge_sets:
      assert isinstance(ms, MergeSet)
      assert isinstance(ms.nodes, set)
      assert isinstance(ms.rationale, str)
      assert isinstance(ms.parameters, dict)
    # Save output to file for inspection
    with open('unit_tests/merge_llm_output.json', 'w', encoding='utf-8') as f:
      json.dump([ms.to_dict() for ms in merge_sets], f, indent=2, ensure_ascii=False)
    # Print JSON output to terminal
    print(json.dumps([ms.to_dict() for ms in merge_sets], indent=2, ensure_ascii=False))

# To run: pytest test_merge_llm.py




