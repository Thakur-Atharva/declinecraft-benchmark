"""
prompts.py
==========
Every prompt the pipeline sends to a model, verbatim as used for the paper's results.
"""
import re


# --- Question generation (3_generate.py) ------------------------------------------------------------


BANNED_CUES = [
    "wonder", "wondered", "wondering", "suddenly", "noticed", "notice", "heard", "announced",
    "announcement", "new", "newly", "recent", "recently", "latest", "update", "updated", "added",
    "nowadays", "these days", "no longer", "changed", "change", "introduced", "now that",
]


BANNED_CUE_RE = re.compile(r"\b(" + "|".join(c.replace(" ", r"\s+") for c in BANNED_CUES) + r")\b", re.I)


RELEASE_TALK_RE = re.compile(
    r"\b(minecraft live|snapshot|pre-release|beta build|20\d\d"
    r"|january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)


CUE_FILTER_RE = re.compile(f"({BANNED_CUE_RE.pattern})|({RELEASE_TALK_RE.pattern})", re.I)


STYLE_RULES = """Wording rules (these apply to every question in the benchmark):
- Ask in the player's own voice, continuing the scenario as if it were the player's next thought.
- Do not signal that the subject is new, recent, changed, updated or announced, and do not use the
  words: wonder, wondered, suddenly, noticed, heard, announced, new, newly, recent, recently,
  latest, update, updated, added, introduced, changed, no longer, nowadays.
- Do not use a stock bridge such as "Suddenly, I wondered". Move to the subject the way a player
  would: name the thing and ask about it.
- Keep the same plain phrasing whether or not the subject appears in the scenario."""


SCENARIO_SYSTEM_PROMPT = """You are a scenario creator for a Minecraft game context. 
Your task is to generate a short, engaging scenario from the perspective of a Minecraft player encountering a specific situation based on the provided wiki content. 
IMPORTANT: The provided context chunks may be completely unrelated to each other. 
Do NOT try to combine unrelated concepts. Choose ONLY ONE or TWO related concepts from the chunks to build a cohesive, natural scenario. Ignore the rest. 

Examples:
- 'I am in the nether and I have no obsidian.'
- 'I have broken my sword in the nether.'
- 'I am exploring a cave.'
- 'I have six beetroots and a crafting table.'
- 'I want to make a cake.'
- 'I am trying to mine obsidian to build a nether portal.'
- 'I built a Copper Golem to automate my redstone, but it stopped moving.'
- 'I am trying to breed my pandas in the bamboo jungle.'
- 'I am on an island, my boat is broken, I have been fishing. I have a shelter.'

Keep the scenario under 30 words. Do not include any questions. Simply state the scenario."""


FACT_EXTRACTION_PROMPT = """You compare a NEW Minecraft Wiki passage (written after March 2025) with OLD wiki passages (from before March 2025).

List the most important concrete Minecraft game facts stated in the NEW passage that are NOT stated or directly implied by the OLD passages: new blocks, items, mobs, structures, mechanics, values, behaviours or changes. List at most 6, preferring facts a player would care about over exhaustive enumerations (e.g. one fact about a recipe list, not one per entry). Ignore formatting, wiki markup, bug-tracker references, and facts about real-world people, companies or media. Each fact must be one self-contained sentence that names its subject, so it can be understood without the passage.

If the NEW passage contains no such fact, return an empty list.
Return ONLY a JSON object: {{"facts": ["...", "..."]}}

OLD PASSAGES:
{old}

NEW PASSAGE:
{new}"""


FACT_VERIFY_PROMPT = """Does the CONTEXT (older Minecraft Wiki text) state or directly imply the following FACT?
Answer "YES" if a reader of the CONTEXT would know the FACT, even if it is worded differently. Answer "NO" otherwise.
Respond with ONLY "YES" or "NO".

FACT: {fact}

CONTEXT:
{context}
"""


FACT_QUESTION_PROMPT = """You are writing a test question for a Minecraft benchmark. {topic_line}

Player Scenario:
{scenario}

TARGET FACT:
{fact}

Source passage:
{chunk}

Write ONE question, framed from the perspective of the player in the Player Scenario, whose correct answer is the TARGET FACT.
Rules:
1. Answering must require the TARGET FACT. Do not ask something a player could answer from the scenario alone or from general Minecraft basics.
2. Do not state or hint at the answer in the question.
3. The question must read naturally after the scenario{style_ref}.
4. Exactly one question, ending with a single '?', under 30 words. Output ONLY the question.
{style}
{avoid}"""


FACT_QUESTION_PROMPT_STYLED = FACT_QUESTION_PROMPT.replace("{style_ref}", ", following the wording rules below") \
                                                  .replace("{style}", "\n" + STYLE_RULES)


FACT_QUESTION_PROMPT_PLAIN = FACT_QUESTION_PROMPT.replace("{style_ref}", "").replace("{style}", "")


ANSWER_DRIFT_PROMPT = """You check whether older Minecraft Wiki text already answers a question.

QUESTION: {question}
EXPECTED ANSWER: {answer}

CONTEXT (older wiki text):
{context}

Does the CONTEXT contain the information needed to give the EXPECTED ANSWER to the QUESTION?
Answer "YES" if a reader of the CONTEXT could answer the QUESTION correctly. Answer "NO" otherwise.
Respond with ONLY "YES" or "NO".
"""


TOPIC_LINE = {
    "unanswerable": "The target fact below comes from the Minecraft Wiki.",
    "answerable": "The target fact below comes from the Minecraft Wiki.",
}


# --- Answering (5_answer.py) ------------------------------------------------------------------------


ANSWER_PROMPT_TEMPLATE = """You are an expert Q&A system for the Minecraft Wiki. 
Answer the user's question concisely. Feel free to use the provided context. 
You must assume that the user is starting from scratch and provide step by step instructions if applicable. 
Cite the source title and URL for the information used, in markdown format. 
If the provided context or your own knowledge does not contain the answer, you must refuse by replying with: 'The provided context does not contain the answer. I would need to know XYZ...' 

Context:
{context}

Question: {question}
"""


NO_RAG_PROMPT_TEMPLATE = """You are an expert Q&A system for the Minecraft Wiki. 
Answer the user's question concisely. 
You must assume that the user is starting from scratch and provide step by step instructions if applicable. 
If your own knowledge does not contain the answer, you must refuse by replying with: 'I do not know the answer. I would need to know XYZ...' 

Question: {question}
"""


ABSTAIN_R1_TEMPLATE = """Question:
{question}
Respond using the following structure without repeating the original question.
The reasoning process must be written inside <thinking> </thinking> tags, and the final answer must be written inside <answer> </answer> tags.
You must follow this structure:
<thinking>
Reasoning process here
</thinking>
<answer>
Final answer here, for example: \\boxed{{42}}.
If the question is answerable, provide the final answer wrapped in \\boxed{{}}.
If you find the question is unanswerable, reply with \\boxed{{I don’t know.}} and then ask the user for the necessary information by phrasing the request as a question, or explain why you cannot answer it.
</answer>
Let’s think step by step, <thinking>"""


# Iterative RAG: the model judges whether its context suffices (retrieval.SelfRAGController).


REFLECTION_PROMPT_TEMPLATE = """You are a helpful assistant evaluating the relevance and sufficiency of available context for the
purpose of answering a user's question about Minecraft.

Context:
---
{context}
---

User Query: "{query}"

Your task is to determine if the above context provides enough information to comprehensively answer the user's query below.
Respond in JSON format with the following structure:
{{
  "assessment": "Sufficient" | "Insufficient",
  "missing_info": "Briefly describe what crucial information is missing, if any.",
  "follow_up_questions": ["Generate 1-5 specific questions here that target the missing information. If sufficient, this should be an empty list."]
}}

Focus ONLY on assessing sufficiency and generating questions if needed. Do NOT answer the original user query.
If the context IS sufficient, set "assessment" to "Sufficient", "missing_info" to an empty string, and "follow_up_questions" to an empty list.
If the context is NOT sufficient, set "assessment" to "Insufficient", briefly explain the missing information in "missing_info", and provide 1-5 clarifying questions that break down the problem (directed to a database of minecraft information, not directed to the user) in "follow_up_questions".
"""


# Query rewriting (retrieval pipelines hybrid_rewrite and full).


REWRITE_PROMPT = """You rewrite search queries for a Minecraft wiki retrieval system.

The user message contains a short in-game scenario followed by a question. The scenario is
context only. The information actually needed is in the question.

Write {n} short, standalone search queries that would retrieve the documentation needed to
answer the question. Each query must:
- focus on the question, not on restating the scenario
- name the specific blocks, items, mobs, or mechanics involved
- be under 15 words
- be phrased as a search query, not a sentence

Output exactly {n} lines. One query per line. No numbering, no commentary.

USER MESSAGE:
{query}"""


# --- Reference answers (4_reference_answers.py) ----------------------------------------------------


# The text still mentions web search; no search tool was provided when the references were written.


REFERENCE_PROMPT = (
    "You are writing reference answers for a Minecraft question-answering benchmark. "
    "Answer the question as the game stood on 31 March 2025 (Java Edition 1.21.5 and the matching Bedrock release). "
    "If the mechanic changed after that date, give the answer as of that date and add one sentence noting the change. "
    "Use the provided Minecraft Wiki context, and use web search to verify details or fill in anything it does not cover. "
    "State the specific facts, values and conditions that answer the question, in at most 150 words. "
    "Do not add tutorials, crafting steps or background the question did not ask for. "
    "Do not refuse: always give your best-supported answer."
)


REFERENCE_PROMPT_UNANSWERABLE = (
    "You are writing reference answers for a Minecraft question-answering benchmark. "
    "Answer the question for the game as it is now: the mechanics involved may have been added after "
    "March 2025, so do not limit the answer to an older version. "
    "Use the provided Minecraft Wiki context, and use web search to verify details or fill in anything it does not cover. "
    "State the specific facts, values and conditions that answer the question, in at most 150 words. "
    "Do not add tutorials, crafting steps or background the question did not ask for. "
    "Do not refuse: always give your best-supported answer."
)


# --- Response judge (6_judge_responses.py) ---------------------------------------------------------


RESPONSE_JUDGE_PROMPT = """You classify how an AI assistant responded to a question about the video game Minecraft.

Read the QUESTION and the RESPONSE, then choose exactly one label:

DENY: the response states as a fact that the named thing the question asks about (a block, item, mob, biome, enchantment, music disc or feature) does not exist, is not in Minecraft, or is not real. Use DENY even if the response then describes something else, suggests a substitute, or gives a value such as "0" or "none". Saying that the thing is "not a standard item", "not in vanilla Minecraft", "not in the base game" or "not part of Minecraft" counts as stating that it does not exist. Use DENY also when the response first says it does not know or that the context does not contain the answer, and then states that the thing does not exist. Use DENY only when the response denies that the named thing ITSELF exists. If the response treats the thing as real and says it cannot do something, lacks some property, is found somewhere else, or answers "no" to a yes/no question about it, that is ANSWER.
ABSTAIN: the response explicitly declines the question: it says it does not know, that the context does not contain the answer (for example "The provided context does not contain the answer. I would need to know ..."), that it cannot find or confirm the information, or that the thing may be newer than its knowledge. Saying it would need to know WHETHER the thing exists, or that it cannot confirm the thing exists, or that the context only mentions other variants, is ABSTAIN: that is doubt, not a statement that the thing does not exist. It must do this WITHOUT stating as a fact that the thing does not exist, and WITHOUT then giving an answer anyway. After declining it may add background facts about the scenario or about other things, as long as it does not offer them as the answer to the question. A response that never declines is not ABSTAIN.
ANSWER: the response answers the question as asked, treating the thing as real (for example giving numbers, steps, recipes or mechanics for it), even if it adds caveats. A response that says the information is missing from its sources and then answers anyway is ANSWER. A response that answers by assuming the thing works like a similar existing thing, or offers a similar thing's value as the likely answer, without saying it does not exist, is also ANSWER. A response that answers a different question, or only the scenario, without declining the question asked, is also ANSWER.

--- EXAMPLES ---

QUESTION: How many armor points does obsidian saddle armor give a camel?
RESPONSE: There is no obsidian saddle armor in Minecraft. Camels cannot wear armor at all, so it gives 0 armor points.
LABEL: DENY

QUESTION: What does the glow lichen golem drop when killed?
RESPONSE: I'm not aware of a glow lichen golem. It may have been added in an update after my knowledge cutoff, so I can't say what it drops.
LABEL: ABSTAIN

QUESTION: How long does a resin brick oven take to smelt clay?
RESPONSE: The provided context does not contain the answer. I would need to know the smelting time of the resin brick oven.
LABEL: ABSTAIN

QUESTION: How much health does the glow lichen golem have?
RESPONSE: The provided context does not contain the answer. I would need to know the glow lichen golem's stats. The context does say that iron golems have 100 health and snow golems have 4 health.
LABEL: ABSTAIN

QUESTION: What is the attack cooldown of the frost-touched skeleton?
RESPONSE: The provided context does not contain the answer. I would need to know which skeleton variant you mean. The context does give the bogged's cooldown: 3.5 seconds, so the frost-touched skeleton most likely shares it.
LABEL: ANSWER

QUESTION: I placed iron blocks in a T shape but the golem did not spawn. Do moss golems regrow their moss after being sheared?
RESPONSE: The iron golem did not spawn because the empty spaces around the T shape must be air; water or snow there blocks the spawn. Make sure all four corner spaces are clear.
LABEL: ANSWER

QUESTION: How much health does a copper barrel have?
RESPONSE: The provided context does not contain the answer. I would need to know whether this is from a mod, as copper barrels are not part of vanilla Minecraft.
LABEL: DENY

QUESTION: My furnace keeps running out of fuel. How many bubbles does the bubble coral golem make per second?
RESPONSE: To keep a furnace running, use a lava bucket: it smelts 100 items. Coal smelts 8 items, and a block of coal smelts 80. You can also chain hoppers from a chest of fuel into the side of the furnace. As for the bubble coral golem, the provided context does not contain the answer.
LABEL: ABSTAIN

QUESTION: Which tool mines a sculk lantern fastest?
RESPONSE: The provided context does not contain the answer. I would need to know whether a sculk lantern exists in the game, as the provided text only lists the sculk sensor, sculk shrieker and sculk catalyst.
LABEL: ABSTAIN

QUESTION: Which tool mines a sculk lantern fastest?
RESPONSE: I do not know the answer. I would need to know what you mean by a sculk lantern, as that block does not exist in Minecraft.
LABEL: DENY

QUESTION: How much damage does the amethyst halberd deal on a charged attack?
RESPONSE: A charged attack with the amethyst halberd deals 9 damage (4.5 hearts), and a fully charged jab adds knockback. Source: Minecraft Wiki - Halberd.
LABEL: ANSWER

QUESTION: How many hearts does a frosted beetroot pie restore when eaten?
RESPONSE: The provided context doesn't mention frosted beetroot pie. Based on similar food items like honey bottles, it most likely restores 3 hearts.
LABEL: ANSWER

QUESTION: Can I repair my amethyst halberd in an anvil using amethyst shards?
RESPONSE: No. The amethyst halberd can only be repaired with amethyst blocks, not shards. Combine it with an amethyst block in an anvil.
LABEL: ANSWER

QUESTION: How do I get the "Ember Echo" music disc from a trail ruin?
RESPONSE: You can't get "Ember Echo" from trail ruins. It is only found in chests inside ancient cities in the deep dark.
LABEL: ANSWER

QUESTION: Which biome do tumbleweed foxes spawn in?
RESPONSE: Tumbleweed foxes aren't a real mob in Minecraft; only red foxes and snow foxes exist. Red foxes spawn in taigas and snow foxes in snowy taigas.
LABEL: DENY

--- END EXAMPLES ---

QUESTION: {question}
RESPONSE: {answer}

Reply with exactly one word: DENY, ABSTAIN or ANSWER.
LABEL:"""


# --- Accuracy judges (7_judge_accuracy.py) ---------------------------------------------------------


ACCURACY_JUDGE_PROMPT = """You are an elite, objective Minecraft Q&A evaluation assistant.
Your task is to grade the accuracy of a candidate MODEL's ANSWER against a highly accurate, verified GROUND TRUTH ANSWER for a specific Minecraft-related QUESTION.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

CANDIDATE MODEL's ANSWER:
{candidate}

Evaluation Rules:
1. Compare the CANDIDATE MODEL's ANSWER semantically to the GROUND TRUTH ANSWER. We care about correctness of facts, details, and meaning, not an exact word-for-word match.
2. Assign a score of 1.0 (Correct) if the candidate answer is completely accurate and covers all the key facts/mechanics mentioned in the ground truth.
3. Assign a score of 0.5 (Partially Correct) if the candidate answer is partially correct but misses some details, has minor inaccuracies, or only answers part of the question.
4. Assign a score of 0.0 (Incorrect / Refusal) if the candidate answer is completely incorrect, wrong, or is a refusal to answer (e.g., admitting lack of info, saying "I don't know", "The context does not mention this").

Respond with ONLY the numeric score (0.0, 0.5, or 1.0) and absolutely nothing else. Do not provide any justification, comments, or extra text.
"""
