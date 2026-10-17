# About Course: Multi-Agent Systems and Protocol Integration on AWS

## The problem this course solves

A single agent degrades as scope grows: it handles one request type well, blurs on the second, and
hallucinates by the third. Prompting harder does not fix this — the fix is a smaller surface per model
plus an architecture that coordinates several of them. This course is the shift from "an agent" to "an
agent system": decomposing a problem into specialized agents, giving them a router, and giving them
shared context.

## Learning outcomes → what each one actually requires

| # | Outcome | Depends on |
|---|---|---|
| 1 | Design multi-agent architectures (hierarchical, flat, hybrid) with clear boundaries, communication paths, tradeoffs | no new AWS service — this is the design layer above everything else |
| 2 | Implement on Bedrock AgentCore + Strands Agents SDK, matching model to role (Nova Lite / Nova Pro / Claude) | Bedrock AgentCore, Strands Agents SDK |
| 3 | Orchestrate and route (sequential, parallel, conditional; content- and LLM-based routing; Saga-style state coordination) | DynamoDB, AgentCore Memory |
| 4 | Build multi-agent RAG with specialized retrievers | Bedrock Knowledge Bases, Amazon S3 Vectors |
| 5 | Deploy to production | Bedrock Guardrails, AgentCore Gateway, CloudWatch + X-Ray, AgentCore Runtime versioning |

## Capstone: NovaMart

Fictional e-commerce company. Role: Enterprise AI Solutions Architect.

**Five agents:**
- Orchestrator — routes incoming work to the right specialist
- Inventory — order lookups
- Policy — policy questions; backed by parallel multi-agent RAG across **three** Bedrock Knowledge Bases
- Refund — refund decisions
- Communication — customer-facing responses

**Production surface:** AgentCore Runtime deployment, Bedrock Guardrails, session memory, CloudWatch +
X-Ray observability.

**What each piece is for, per the course framing:**
- Bedrock Guardrails — makes refusal predictable when the system should not take on a piece of work
- AgentCore Gateway — handles the handoff when agents pass control back and forth
- DynamoDB — shared memory across a long interaction, so agents do not lose the thread

# Module 1: Sequential Multi-Agent Coordinator Pattern

## Why a monolithic agent breaks down

One employee handling customer support, server diagnostics, and payroll at once would struggle, make
mistakes, and eventually fail. A monolithic agent degrades the same way as scope grows — not because
the model is weak, but because the *task itself* is no longer one job. The fix is architectural:
decompose the workflow into distinct roles, same as a hospital splits intake across a front-desk
receptionist, a triage nurse, and a scheduling administrator.

**Why specialized agents actually help**, per the module's framing:
- A single focused responsibility means a tightly customized prompt and a tightly scoped toolset —
  nothing to confuse the model about which job it's doing.
- Smaller responsibility surfaces reduce hallucination and logic errors, because there is less for the
  model to get wrong in a single turn.
- Isolated steps are each independently debuggable, testable, and optimizable — a failure localizes to
  one agent instead of hiding somewhere in one giant prompt.

## The pattern: sequential coordinator

A **sequential multi-agent architecture** chains agents so the output of one becomes the input of the
next. The **coordinator pattern** is the piece that manages that handoff — and the key design decision
in this module is that **the coordinator is not an agent**. It has no model and no system prompt; it is
plain Python that calls each agent in a fixed order and threads data between them. All the intelligence
lives inside the three agents; the coordinator just follows a fixed recipe.

```
Patient Complaint
      │
┌─────▼──────┐
│ Coordinator │   plain Python — no model, no prompt
└──┬───┬───┬─┘
   │   │   │
   ▼   │   │
Symptom       Agent 1: complaint text → conditions + severities
Analyzer
   │
   ▼
Urgency       Agent 2: severities → urgency level (urgent/standard/routine)
Classifier
   │
   ▼
Appointment   Agent 3: urgency level → booked time slot
Scheduler
   │
   ▼
Triage Result
```

## Worked example: 3-agent healthcare triage

Demo file: `lab_1/Demo/healthcare_triage.py`. A patient complaint runs through three agents in strict
sequence — SymptomAnalyzer → UrgencyClassifier → AppointmentScheduler — each a separate `Agent`
instance with its own model, system prompt, and single tool.

**Sample data backing the demo:**
- `PATIENT_COMPLAINTS` — three patients: Alice (chest pain, shortness of breath, dizziness), Bob (mild
  headache, runny nose), Carol (swollen, twisted ankle).
- `AVAILABLE_SLOTS` — maps urgency level (`urgent` / `standard` / `routine`) to a list of time slots.
- `SYMPTOM_CONDITIONS` — a simulated knowledge base mapping symptom keywords to a condition and a
  severity (`high` / `medium` / `low`). The matching is **plain keyword lookup inside the tool**, not
  something the LLM infers — the model never invents medical knowledge, it just calls the tool and
  relays structured output.

### The shared-state trick: capture tool results directly

Before any agent is defined, the file sets up one shared dict:

``` Python
# Shared dict for capturing tool results directly.
# Why: LLMs may rephrase or modify tool output in their text response,
# so we capture the raw tool result here for reliable downstream use.
_tool_results = {}
```

Every tool's last line writes its JSON result into this dict (`_tool_results["symptoms"] = result`,
`["urgency"]`, `["booking"]`) before returning. **This is the contract the coordinator depends on.** The
coordinator never parses the agent's natural-language reply — LLM text is paraphrased and unreliable to
parse — it reads the structured value straight out of `_tool_results`. The tool's return value still
matters because the agent's next reasoning step and the final printed response depend on it, but the
dict is the coordinator's actual data channel.

`_tool_results.clear()` runs at the top of the coordinator for every new patient, specifically to stop
one patient's data bleeding into the next run.

### One builder shape, repeated three times

Every agent is built by a `build_*()` function returning a fresh `Agent`, and all three follow the exact
same shape: a `BedrockModel` at `temperature=0.0`, one `@tool`-decorated function that does a lookup and
writes into `_tool_results`, and a system prompt that both states the single job and explicitly forbids
the other two jobs.

``` Python
def build_symptom_analyzer() -> Agent:
    model = BedrockModel(model_id=MODEL_ID, region_name=AWS_REGION, temperature=0.0)

    @tool
    def lookup_symptoms(complaint_text: str) -> str:
        """Parse a patient complaint and match symptoms against the knowledge base."""
        # ... match complaint_text against SYMPTOM_CONDITIONS keywords ...
        result = json.dumps({"matched_symptoms": matches, "total_matches": len(matches)})
        _tool_results["symptoms"] = result   # capture for the coordinator
        return result

    system_prompt = """You are a Symptom Analyzer agent. Your ONLY job is symptom analysis.
Call the lookup_symptoms tool with the patient's complaint text.
After the tool returns, output ONLY the raw JSON result. Do not classify urgency or book appointments."""

    return Agent(model=model, system_prompt=system_prompt, tools=[lookup_symptoms])
```

`build_urgency_classifier` and `build_appointment_scheduler` are the same shape with a different tool
and a different `_tool_results` key (`"urgency"`, `"booking"`). The urgency rule is a plain if/elif
ladder inside the tool, not a model judgment: any `high` severity → `urgent`; else any `medium` →
`standard`; else `routine`.

> **Note:** every system prompt ends with an explicit negative instruction — "Do not classify urgency
> or book appointments." Stating the job positively was not treated as sufficient; the boundary is
> spelled out too.

### The coordinator itself

``` Python
def run_triage_pipeline(patient: dict) -> dict:
    _tool_results.clear()  # reset between patients to avoid stale data

    run_agent_with_retry(build_symptom_analyzer,
        f"Analyze these symptoms: {patient['complaint']}")
    symptom_json = json.loads(_tool_results["symptoms"])

    run_agent_with_retry(build_urgency_classifier,
        f"Classify urgency for this symptom analysis: {json.dumps(symptom_json)}")
    urgency_level = json.loads(_tool_results["urgency"])["urgency"]

    run_agent_with_retry(build_appointment_scheduler,
        f"Book an appointment for patient_id={patient['patient_id']} with urgency_level={urgency_level}")
    return json.loads(_tool_results["booking"])
```

Three blocks, one per agent, same shape each time: call the builder through a retry wrapper, read the
result back out of `_tool_results`, pass the relevant piece forward as the next agent's prompt text.
`run_agent_with_retry` (defined elsewhere in the file) wraps the call in exponential backoff — 1s, 2s,
4s — for transient Bedrock throttling, and is explicitly called out as a demo convenience that would
move to a shared `utils.py` in production, not something to study as the core pattern.

**A fresh `Agent` instance is built per request, every single time** (`agent = agent_builder()` inside
the retry wrapper, called fresh per patient). This is deliberate, not incidental: reusing one agent
instance across patients would let one patient's conversation history leak into the next one's context.

### Demo output

| Patient | Complaint | Urgency | Slot |
|---|---|---|---|
| Alice | chest pain, shortness of breath, dizziness | urgent | 09:00 AM (Emergency) |
| Bob | mild headache, runny nose | routine | 02:00 PM |
| Carol | swollen, twisted ankle | standard | 10:30 AM |

> **Note:** the recorded walkthrough states Carol's slot as "10:30 PM," but `AVAILABLE_SLOTS` in the
> actual file and the demo's own README both say **10:30 AM** — corrected here against the source file
> rather than the narration.

> **Note:** the lesson text calls the model generically "Claude Bedrock" / "Claude 3 Sonnet" in the
> header comment, but the file's actual default is
> `MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"` (overridable via the `MODEL_ID` env var).
> The header comment and the real default disagree with each other inside the same file — treat the
> `MODEL_ID` constant as the source of truth, not either prose description.

## Four things to carry forward

1. **Each agent has exactly one job** — one model, one prompt, one tool. Not "mostly one job."
2. **The coordinator is plain Python, not an LLM.** No model, no prompt, no reasoning — a fixed
   sequence of calls.
3. **Data flows between agents as JSON**, captured directly from tool output — never by parsing the
   model's prose reply.
4. **A fresh agent instance per request** avoids context bleeding between unrelated runs.

## Key terms

- **Coordinator Pattern** — a central function or agent that manages data flow and task execution
  across specialized agents. In this module's version, the coordinator is specifically *not* an agent.
- **Sequential Multi-Agent Architecture** — agents linked in a chain; one agent's output is the next
  agent's input.
- **Single-Responsibility Agent** — an agent scoped to exactly one task, to reduce complexity and
  improve reliability.

## Where this generalizes

The module's suggested transfer case: a support-ticket pipeline with the same three-stage shape —
an **extractor** agent pulls the core issue out of a noisy email, a **classifier** agent assigns a
priority level to the extracted text, and a **response** agent drafts a reply from the classified
result. Same coordinator pattern, same "fresh agent per request" rule, same "read structured output,
not prose" rule — different domain.

# Module 2: Multi-Model Multi-Agent Systems

## What changes from Module 1

Module 1's three agents all used the same `MODEL_ID`. This module keeps the exact same three-agent
coordinator shape — builder function, `@tool`, shared dict, fresh instance per run — and changes
exactly one thing: **each agent now gets a different Bedrock model, chosen for what that agent's job
actually needs.** The architecture lesson is separate from the model-selection lesson, and this module
isolates the second one by holding the first one constant.

## The idea: match the model to the task, not the other way around

Not every step in a pipeline needs the most powerful model. The guiding split used throughout:

| Task shape | Model | Why |
|---|---|---|
| Simple classification (pick one of a few labels) | **Nova Lite** — fastest, cheapest | A lightweight model is not a compromise here; it's the correct tool. Nothing about the task benefits from more reasoning power. |
| Deep reasoning (diagnose a cause, recommend an action) | **Claude** — most capable | This is the step that actually needs the expensive model's judgment. Spend the budget where it buys something. |
| Natural-sounding drafting (customer- or team-facing text) | **Nova Pro** — balanced | Good enough writing quality without paying Claude's price or latency for prose generation. |

The one-line version: **fast and cheap where you can, powerful only where you must.**

## Worked example: 3-agent incident response

Demo file: `lab_2/Demo/incident_response.py`. An infrastructure alert runs through the same
three-stage shape as Module 1's triage pipeline — Alert Router → Root Cause Analyzer → Status
Drafter — but each stage is backed by a different model:

```
Infrastructure Alert (incident_id)
      │
┌─────▼──────────────┐
│  Python orchestrator │   same role as Module 1's coordinator — no model, no prompt
└──┬────────┬────────┬┘
   │        │        │
   ▼        │        │
Alert         Agent 1: incident_id → severity (critical/warning/info)
Router        Model: Nova Lite — fast, cheap, deterministic (temp 0.0)
(Nova Lite)
   │
   ▼
Root Cause    Agent 2: incident data → probable cause + runbook step + ETA
Analyzer      Model: Claude — most capable, the actual reasoning task (temp 0.1)
(Claude)
   │
   ▼
Status        Agent 3: severity + incident data → 4-line status update
Drafter       Model: Nova Pro — balanced, good enough prose, cheaper/faster than Claude (temp 0.3)
(Nova Pro)
   │
   ▼
Final Status Update
```

**Sample data backing the demo:**
- `INCIDENTS` — three incidents: INC-001 critical (CPU at 98%, memory at 94%, 47 5xx errors),
  INC-002 warning (disk at 82%, full in 9 days), INC-003 info (deployment completed, 0 canary errors).
- `SEVERITY_KEYWORDS` — same keyword-matching idea as Module 1's `SYMPTOM_CONDITIONS`: the tool, not
  the model, decides severity by scanning the incident message for phrases like `"5xx errors"`
  (critical), `"disk usage at 8"` (warning), `"completed"` (info).
- `KNOWN_ISSUES` — a runbook table keyed by category (`high_cpu`, `disk_usage`, `deployment`), each
  entry giving a probable cause, common causes, runbook steps, and an estimated resolution time.
- `STATUS_TEMPLATES` — maps severity to an audience and tone (e.g. critical → "Engineering +
  Management + On-call", "Urgent, concise, action-oriented").

### Three model IDs, read from the environment

``` Python
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")                     # fast
CLAUDE_MODEL    = os.environ.get("CLAUDE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")  # deep
NOVA_PRO_MODEL  = os.environ.get("NOVA_PRO_MODEL", "amazon.nova-pro-v1:0")                       # balanced
```

Reading each model ID from an environment variable (with a hardcoded default) means swapping a model
is an `.env` edit, not a code change. This is worth pausing on, because it's the same design principle
that distinguishes AgentCore harness from a hand-built Runtime deployment — just applied at a much
smaller scale.

**The principle: separate *what* an agent is from *how* it's built, and put the "what" in config.**
Anything you might reasonably want to change later without re-reasoning about the code — which model,
which region, which prompt — should live outside the code path, where editing it is cheap, reviewable,
and can't accidentally break logic. The code then reads the config at startup and does the same thing
every time, regardless of what the config said.

**At the big scale — the harness vs. Runtime split.** AgentCore harness takes this principle all the way
up: the *entire agent* is configuration. Model, system prompt, tools, memory, limits — all declared, none
coded. Swapping the model is a config field edit, and you never redeploy a container because there is
no container you own. A hand-built Runtime deployment is the opposite end: the orchestration loop is
code you wrote, so changing the model means editing that code, rebuilding the ARM64 image, pushing to
ECR, and redeploying.

**At the small scale — this file.** `incident_response.py` is firmly on the Runtime side of that split:
it is code you own, with an orchestration loop you wrote. But *inside* that code, the author still
carved out the one knob most likely to change — which model backs each agent — and moved it to config:

``` Python
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")
```

So the builder function `build_routing_agent()` never hardcodes a model string. It reads a variable.
If a cheaper or newer Nova model ships next month, nobody touches `build_routing_agent()` — they change
one line in `.env` and rerun. The agent's *logic* (call `classify_alert`, report two lines) is untouched;
only its *identity* (which model does the thinking) moved.

**Why "one level down" is the right way to see it.** The harness applies config-not-code to the whole
agent — the unit of configuration is "an agent." This file applies the identical principle to one field
inside one agent's builder — the unit of configuration is "a model ID." Same instinct, smaller target.
The useful takeaway isn't that these two things are literally the same mechanism (they aren't — one is
a managed AWS abstraction, the other is `os.environ.get`); it's that **the same question drives both
decisions**: *what here will I want to change without wanting to think about the code again?* Put that
thing in config. Everything else can stay in code.

### The cross-model handoff: same shared-dict pattern as Module 1, renamed

``` Python
# KEY PATTERN: Shared cache so Python can read tool outputs across agents.
classification_cache = {}
```

This is Module 1's `_tool_results` dict under a new name and a narrower job — it only needs to carry
the severity classification forward, since that's the one piece of Agent 1's output that Agent 3's
prompt text needs. The principle is identical: **the tool writes, the orchestrator reads** — never
parse an LLM's prose reply when a structured value is sitting right there in a dict.

### One builder shape, three different models

All three agents still share Module 1's exact builder shape — `BedrockModel` + system prompt +
`@tool` — and differ only in `model_id` and `temperature`:

``` Python
def build_routing_agent() -> Agent:
    # Nova Lite at temp 0.0 — fast, deterministic classification, no creativity needed
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)

    system_prompt = """You are an alert routing agent. Your ONLY job:
1. Call classify_alert with the incident_id provided by the user
2. Report the result in exactly 2 lines:
   Severity: <CRITICAL|WARNING|INFO>
   Service: <affected service name>
Do NOT add any other commentary."""

    @tool
    def classify_alert(incident_id: str) -> str:
        # ... match incident message against SEVERITY_KEYWORDS ...
        result = {"incident_id": incident_id, "severity": severity, ...}
        classification_cache[incident_id] = result   # captured for the orchestrator
        return json.dumps(result, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[classify_alert])
```

The other two builders follow the same shape with the temperature chosen to match the job, not left
at a single default across all three agents the way Module 1 did:

- **`build_analysis_agent`** — Claude at `temperature=0.1`. Its `investigate_incident` tool joins the
  incident to the `KNOWN_ISSUES` runbook table and returns the full lookup as JSON; the system prompt
  then forces Claude to compress that into exactly 3 lines (`Root Cause` / `Action` / `ETA`) rather
  than repeating the raw JSON back. The near-zero temperature keeps the reasoning tight while still
  being nonzero, since root-cause synthesis is prose generation, not pure classification.
- **`build_status_agent`** — Nova Pro at `temperature=0.3`, the highest of the three. Its
  `draft_status_components` tool returns the incident data plus the matched `STATUS_TEMPLATES` entry
  (audience + tone); the system prompt caps the output at 4 lines and tells the model to match tone to
  severity. The higher temperature is deliberate — some variation is fine, even desirable, for
  natural-sounding team communication, where Agent 1's classification needed none at all.

> **Note:** temperature itself is scaled to the task across all three agents (`0.0` → `0.1` → `0.3`),
> not just the model choice. The pattern is "match every tunable to the job," not only "pick the right
> model and leave everything else as-is."

### Orchestrating the pipeline

``` Python
for incident in INCIDENTS:
    inc_id = incident["id"]

    routing_agent = build_routing_agent()
    routing_result = routing_agent(f"Classify incident {inc_id}")
    severity = classification_cache.get(inc_id, {}).get("severity", "info")

    analysis_agent = build_analysis_agent()
    analysis_result = analysis_agent(f"Investigate incident {inc_id}")

    status_agent = build_status_agent()
    status_result = status_agent(
        f"Draft a status update for incident {inc_id}. The severity is {severity}."
    )
```

Note what does **not** change from Agent 2 to Agent 3: the analysis agent's result is read from its
return value (then cleaned of `<thinking>` tags), while only the severity — the one value genuinely
needed downstream — comes from the shared dict. Not every piece of inter-agent data has to go through
the cache; only the piece the next prompt actually depends on structurally.

### Demo output

| Incident | Severity (Nova Lite) | Root cause (Claude) | Runbook action | Status drafted (Nova Pro) |
|---|---|---|---|---|
| INC-001 | critical | memory leak or traffic spike | scale horizontally | urgent, audience: Engineering + Management + On-call |
| INC-002 | warning | log accumulation without rotation | enable log rotation | informative, preventive, audience: Engineering |
| INC-003 | info | normal deployment activity | monitor canary metrics | brief, positive, audience: Engineering |

**Latency, as measured and printed by the demo itself:** Nova Lite fastest, Nova Pro in between,
Claude slowest, across all three incidents — the trade matches the model-selection rationale exactly:
you pay more latency only on the step that needed the stronger model.

### Cost projection: the actual math, checked against the file

The demo computes a cost projection rather than quoting a fixed number — it uses Bedrock's per-1M-token
pricing (Nova Lite, Nova Pro, Claude 3 Sonnet; explicitly marked in the file as *indicative, April 2026
reference pricing, check the live pricing page before sizing anything real*) against an assumed 200
input / 100 output tokens per agent call, then projects to 10,000 incidents/day:

``` Python
pricing = {
    "Nova Lite":        (0.06, 0.24),    # $ per 1M tokens, (input, output)
    "Nova Pro":         (0.80, 3.20),
    "Claude 3 Sonnet":  (3.00, 15.00),
}

def dollars_per_call(model_name: str) -> float:
    inp, out = pricing[model_name]
    return (tokens_in_per_call * inp + tokens_out_per_call * out) / 1_000_000

mix_per_incident = (nova_lite_cost + claude_cost + nova_pro_cost)   # one call to each
claude_only_per_incident = 3 * claude_cost                          # all 3 steps on Claude
savings_pct = (claude_daily - mix_daily) / claude_daily * 100
```

Working it through with the file's own numbers: Nova Lite costs \$0.000036/call, Nova Pro
\$0.00048/call, Claude \$0.0021/call (at 200 in / 100 out tokens). So the multi-model mix costs
**\$0.002616 per incident**, against a Claude-for-everything baseline of **\$0.0063 per incident**
(3 × the Claude figure) — a reduction of **≈58–59%**, not a fixed number baked into the code.

> **Note:** the recorded walkthrough and the lesson-text summary both describe this as "roughly 80%,"
> but recomputing the formula from the file's own `pricing` dict and `tokens_in_per_call` /
> `tokens_out_per_call` constants gives **≈58.5%**, not 80%. Possibly the spoken number referred to a
> different run with different assumed token counts, or was simply imprecise — flagging rather than
> silently adopting either figure. The mechanism ("mix beats Claude-for-everything, substantially") is
> correct either way; only the specific percentage is in question. The file itself is explicit that
> these are indicative numbers to be checked against live pricing, so re-deriving the percentage from
> whatever pricing is current at the time is the right move before quoting it anywhere real.

## Four things to carry forward

1. **Model choice is a per-agent decision, not a per-pipeline one.** The same pipeline can and should
   mix models when the steps have genuinely different demands.
2. **Classification and routing rarely need the frontier model.** Keyword-matching-backed, low/zero
   temperature, cheapest model is usually correct for "pick one of a few labels."
3. **Reserve the expensive model for the step that is actually reasoning** — synthesizing a cause from
   evidence, not relaying a lookup.
4. **Temperature is tuned per task too**, not just model choice: deterministic for classification,
   slightly open for structured analysis, more open for natural-sounding prose.

## Key terms

- **Multi-model architecture** — a multi-agent system where different agents use different underlying
  foundation models, each chosen for that agent's specific task demands rather than using one model
  everywhere.
- **Model-task matching** — the practice of pairing task difficulty/type (classification vs. reasoning
  vs. drafting) to model capability/cost tier, rather than defaulting to the most capable model for
  every step.

# Module 3: Parallelization

## Why run agents side by side

Modules 1 and 2 ran agents strictly one after another — each stage waited for the previous one to
finish because each stage *needed* the previous one's output. Parallelization is for the case where
that dependency isn't there: multiple agents work on different parts of a task, or even the same task,
**at the same time**.

The obvious benefit is speed, and the module is explicit that speed is **not the main point**. Running
several agents concurrently also produces **diverse perspectives or solutions** on the same problem,
which can yield a richer or more dependable result than any single agent would. Parallelism is a
quality tool as much as a latency tool — a point that becomes concrete in the "identical tasks for
voting" decomposition strategy and the "synthesizer LLM" aggregation strategy below.

## The pattern: scatter-gather

The shape is borrowed directly from distributed computing's **scatter-gather** pattern: a problem is
*scattered* to multiple workers, each works independently, and their findings are *gathered* back into
one result. Three pieces, always:

1. **A way to split the main task** into subtasks.
2. **Individual agents that each perform a subtask** — concurrently.
3. **A method to aggregate their outputs** into a single coherent result.

```
            Main Task
                │
        ┌───────┼───────┐        split (decomposition)
        ▼       ▼       ▼
     Agent A  Agent B  Agent C   run CONCURRENTLY — none waits for another
        │       │       │
        └───────┼───────┘        gather (aggregation)
                ▼
          Final Result
```

Contrast with Modules 1–2, where the arrows ran *down a chain* and each agent blocked on the one
above it. Here they fan out and fan back in.

## The golden rule: subtasks must be independent

This is the prerequisite everything else rests on. **Agent A working on subtask A must not need to
wait for Agent B to finish subtask B.** If one subtask's output is a direct input to another, those two
are *not* parallelizable — that dependent portion wants a sequential pattern (the module names
**prompt chaining**, which is what Modules 1 and 2 were actually doing) instead.

The practical consequence: deciding whether to parallelize is not a performance decision you make
after the fact. It is a *dependency analysis* you do first. Draw the data flow; wherever an arrow goes
from one subtask's output into another's input, you have a sequential edge. Only the subtasks with no
such edges between them can run side by side. A real system usually ends up **hybrid** — parallel
fan-out for the independent parts, sequential chaining for the dependent ones.

## Decomposition: three ways to split a task

How you split determines what you can run in parallel. Three strategies, each fitting a different
shape of problem:

| Strategy | When to use it | Example | What's actually parallel |
|---|---|---|---|
| **Sectioning / sharding** | A large, *divisible* input | Each chapter of a book summarized by a different agent at once | Chunks of one input, same operation on each |
| **Aspect-based decomposition** | One subject needing analysis along several *independent* dimensions | Product review: one agent on technical specs, one on user sentiment, one on competitor pricing | Different *questions* about the same single input |
| **Identical tasks for diversity / voting** | You want a range of answers, or a more reliable single answer | Run the same classification N times (varied prompts or models), then vote | The *same* operation repeated — parallelism for robustness, not throughput |

The third row is the one that makes "not just about speed" literal: you are spending more compute, not
less, on purpose, to buy reliability or creative range.

> **Note:** the module's hands-on exercise (see below) uses **aspect-based decomposition**
> specifically — several agents each analyzing a distinct facet of one subject, then synthesized.

## Aggregation: four ways to gather the results

Once the parallel work finishes, something has to combine it. Four strategies, roughly in order of
sophistication:

| Strategy | How it works | Fits which decomposition |
|---|---|---|
| **Concatenation** | Just join the outputs end to end — append chapter summaries into one document | Sectioning / sharding |
| **Comparison and selection** | Evaluate multiple candidate solutions against predefined criteria, keep the best one | Identical tasks (diversity) |
| **Voting / majority rule** | When agents independently answer the same question, take the most frequent answer as final — a consensus mechanism that boosts accuracy | Identical tasks (voting) |
| **Synthesizer / aggregator LLM** | A dedicated LLM takes the diverse outputs and weaves them into one coherent, more nuanced response | Aspect-based decomposition |

The pairing in the right-hand column isn't a strict rule, but it's the natural fit: how you split
largely decides how you have to rejoin. Sharded chunks concatenate; multiple votes get counted;
independent *aspects* of one subject need something that can actually reason about how they relate —
which is why the synthesizer is itself an LLM, not plain Python.

> **Note:** the synthesizer is the one aggregation strategy that reintroduces an LLM into the
> coordination layer. Modules 1 and 2 kept the coordinator as plain Python precisely to avoid parsing
> prose. The synthesizer is a deliberate exception: when the parallel outputs are themselves prose
> analyses of different facets, no dict lookup can combine them — only another model can. The
> trade-off is that you've put a model back in the critical path, with everything that implies about
> cost, latency, and non-determinism.

## The hands-on exercise, as described

The module closes by naming the skills its exercise builds. No `lab_3/` demo file exists in this repo
yet, so the following is recorded from the module's own description, **not verified against source
code** — treat the implementation details as what the exercise is *stated* to involve:

- **Aspect-based decomposition** of a complex analytical task into independent subtasks.
- **Python's `threading` module** — creating, starting, and joining threads to run agents
  concurrently.
- **A shared dictionary** as the strategy for collecting results from multiple threads — the same
  "tool writes, coordinator reads" device as `_tool_results` (Module 1) and `classification_cache`
  (Module 2), now with the added wrinkle that several threads write into it at once.
- **Specialized system prompts** defining each agent's distinct analytical role.
- **A synthesizer agent** whose sole job is to take the varied inputs from the parallel agents and
  produce one coherent, higher-level output — the "synthesizer LLM" aggregation strategy, made into a
  concrete agent.

> **Note:** the shared-dictionary-across-threads detail is worth revisiting once the exercise code
> exists. A plain `dict` written from multiple threads works in CPython in practice because the GIL
> serializes individual dict operations, but that is an implementation detail rather than a guarantee,
> and distinct keys per thread (one key per aspect) sidestep the question entirely. Check which the
> exercise actually relies on before generalizing the pattern.

## Four things to carry forward

1. **Independence first, parallelism second.** Do the dependency analysis before deciding anything
   runs concurrently. Dependent subtasks chain; independent ones fan out.
2. **Parallelism buys quality, not only speed.** Running the same task N ways and voting, or
   analyzing N aspects and synthesizing, spends *more* compute to get a *better* answer.
3. **How you split decides how you rejoin.** Shards concatenate, votes get counted, independent
   aspects need a synthesizer.
4. **The synthesizer puts a model back in the coordination layer.** That's a deliberate exception to
   Modules 1–2's plain-Python coordinator, made because prose analyses can't be combined by a dict
   lookup — and it carries the cost, latency, and non-determinism that implies.

## Key terms

- **Parallelization pattern** — multiple agents working on different parts of a task, or the same
  task, concurrently, with results consolidated afterward.
- **Scatter-gather** — the distributed-computing pattern this borrows from: scatter a problem to
  workers, gather their results.
- **Subtask independence** — the prerequisite for parallelization: no subtask's output is another's
  input.
- **Prompt chaining** — the sequential alternative for dependent subtasks; what Modules 1 and 2
  implemented.
- **Sectioning / sharding**, **aspect-based decomposition**, **identical tasks** — the three
  decomposition strategies.
- **Concatenation**, **comparison and selection**, **voting / majority rule**, **synthesizer LLM** —
  the four aggregation strategies.