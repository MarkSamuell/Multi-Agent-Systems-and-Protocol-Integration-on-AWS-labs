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

# Module 3 Lab: Parallel Document Analysis

## Prerequisite: Python's `concurrent.futures` in five concepts

The lab uses two names from the standard library's `concurrent.futures` module, and understanding
what they actually do makes the lab code read as obvious rather than magical. Everything below is from
the official docs (`docs.python.org/3/library/concurrent.futures.html`).

**1. The module's one job: run callables asynchronously behind a uniform interface.** You hand it a
function, it runs the function somewhere else (a thread, a process, or — new in 3.14 — a separate
interpreter), and hands you back a *ticket* you can cash in later. All three backends implement the
same abstract `Executor` interface, so the calling code looks identical regardless of which one you
picked.

**2. `ThreadPoolExecutor(max_workers=N)` — a pool of at most N threads.** You `submit()` callables to
it; it queues them and runs up to N at once, reusing idle threads rather than spawning fresh ones per
task. Use it as a context manager (`with ... as executor:`) so it shuts down cleanly — on exit it waits
for every pending future to finish, equivalent to `shutdown(wait=True)`.

Why threads, and not processes, for LLM agents: a Bedrock call is **network I/O**. The thread spends
almost all its time blocked waiting for the API to respond, during which Python's GIL is *released* and
other threads run. Threads give you real overlap for I/O-bound work. They would give you nothing for
CPU-bound work, where the GIL serializes execution — that's what `ProcessPoolExecutor` is for. The docs
make this explicit: the default `max_workers` is `min(32, cpu_count + 4)`, deliberately *more* than the
CPU count, "assuming that ThreadPoolExecutor is often used to overlap I/O instead of CPU work." Three
agents waiting on three Bedrock responses is the textbook case.

**3. `submit(fn, *args)` returns a `Future` — a placeholder for a result that doesn't exist yet.** The
call returns *immediately*; the work happens in the background. A `Future` has a small, important API:
`.result()` blocks until the value is ready and returns it (or **re-raises any exception the callable
raised** — this is how errors in worker threads reach you); `.done()` tells you whether it's finished
without blocking; `.exception()` returns the raised exception if there was one.

**4. `as_completed(futures)` — an iterator that yields futures as each one finishes, in completion
order, not submission order.** This is the piece that makes the lab's timing honest. If you instead
looped `for f in futures: f.result()`, you'd block on the *first* future even if the third finished
seconds ago. `as_completed` hands you whichever one is ready, so the slowest task gates the whole phase
but nothing else waits unnecessarily. Any future already finished before you call it is yielded first.

**5. The dict-of-futures idiom — the exact pattern the lab uses, lifted straight from the docs'
own example.** This one is easier to understand by tracing a run than by reading the code, so here is
one, with made-up durations: security finishes in 2s, cost in 3s, scalability in 4s.

**Step 1 — build the dict. All three `submit()` calls return instantly; nothing has finished yet.**

``` Python
futures = {
    executor.submit(run_security): "security",
    executor.submit(run_scalability): "scalability",
    executor.submit(run_cost): "cost",
}
```

Each `submit()` starts a task in the background and returns a `Future` — a placeholder object, not a
value. So right after this line runs, `futures` holds **three Future objects as keys, three strings as
values**, and all three tasks are still running:

```
futures = {
    <Future 0x1a... pending>: "security",
    <Future 0x1b... pending>: "scalability",
    <Future 0x1c... pending>: "cost",
}
```

The dict is deliberately "backwards" — you'd normally write `{"security": future}`. It's inverted
because of what comes next: you'll be *handed* a Future and need to find out which task it was, so the
Future has to be the thing you look up *by*.

**Step 2 — `as_completed(futures)` yields each Future the moment it finishes.** Passing a dict here
iterates its keys, i.e. the three Futures. The iterator then blocks until *any* one of them completes,
yields it, and waits for the next. With our durations, the loop body runs three times, in this order:

| Time | What `as_completed` yields | `futures[future]` | `future.result()` |
|---|---|---|---|
| ~2s | the security Future | `"security"` | `2.0` |
| ~3s | the cost Future | `"cost"` | `3.0` |
| ~4s | the scalability Future | `"scalability"` | `4.0` |

Note the order: **security, cost, scalability** — completion order, which is *not* the order they were
submitted (security, scalability, cost). That reordering is the whole reason the dict exists. If the
loop just got "a Future that finished," it would have no idea whether that was the security run or the
cost run. `futures[future]` answers that: it uses the Future it was just handed as a key, and gets back
the label stored at submission time.

**Step 3 — `future.result()` gives the actual value.** `run_agent_with_retry` returns elapsed seconds
(`-> float`), so each `.result()` is a number. Because `as_completed` only yields *finished* Futures,
`.result()` returns immediately here — no blocking.

**The final output.** After the loop, `timings` is:

``` Python
{"security": 2.0, "cost": 3.0, "scalability": 4.0}
```

Insertion order reflects completion order (security, cost, scalability). The *contents* are the same as
if you'd run the three sequentially — the dict is a plain dict of label → seconds either way. What the
idiom bought you is that building it took ~4s (the slowest task) instead of ~9s (the sum), and you
never had to guess which number belonged to which agent.

**Why not just loop over the dict normally?** You could write:

``` Python
for future, label in futures.items():
    timings[label] = future.result()      # blocks on EACH future in submission order
```

This produces the identical final dict — but `future.result()` blocks until *that specific* Future is
done. The first iteration waits ~2s for security (fine), the second waits for scalability to reach ~4s
even though cost finished at ~3s and is sitting there ready. Total wall time is still ~4s, so for a
"wait for everything then read it all" case this is actually fine. `as_completed` matters when you
want to *act on each result as it arrives* — print a progress line, kick off a dependent step, bail out
early on a failure — rather than wait for the slowest one before touching any of them. The lab uses it
mainly as the idiomatic form; the two produce the same `timings` here.

> **Note:** one deadlock the docs call out is directly relevant to agent design: a callable submitted
> to the pool must never *wait on another future from the same pool* if the pool is small enough that
> all workers could end up waiting. With `max_workers=3` and three specialists that never call each
> other, this can't happen here — but it's exactly the trap you'd hit if you tried to make one
> specialist consume another's output inside the pool. That's a dependency, and dependencies belong
> *outside* the parallel phase (which is where the synthesizer lives).

## The lab: three specialists in parallel, one synthesizer after

Demo file: `lab_3/Demo/document_analysis.py`. The scenario is a platform team's launch review: before
a system goes live, someone checks its security, its scalability, and its cost. Done by hand, those are
three sequential reviews. Here, three specialist agents do them **concurrently**, then a synthesizer
agent makes the launch decision.

```
          System Design Document
                   │
      ┌────────────┼────────────┐    ← ThreadPoolExecutor(max_workers=3)
      ▼            ▼            ▼       all three run AT ONCE
   Security    Scalability    Cost
   Reviewer     Reviewer    Reviewer
  (Nova Lite)   (Claude)   (Nova Pro)
      │            │            │
      └────────────┼────────────┘    ← as_completed gathers all three
                   ▼
          Synthesizer Agent (Claude)  ← runs AFTER all three, reads their caches
                   │
                   ▼
     Launch Decision: APPROVE | APPROVE-WITH-CONDITIONS | BLOCK
```

This is Module 3's pattern made concrete, and it's **aspect-based decomposition** specifically: three
agents analyze the *same* document along three *independent* dimensions. The independence is real —
the security reviewer never needs the scalability findings, the cost reviewer never needs the security
findings — which is the golden rule from Module 3 satisfied, and therefore what licenses running them
in parallel at all.

**Why total time is the max, not the sum.** If the three specialists take 2s, 4s, and 3s, the parallel
phase takes 4s (the slowest one), not 9s. Modules 1 and 2 paid the sum because each stage genuinely
waited on the previous one. Here nothing waits, so you pay only for the longest.

### The two documents

- **DOC-001, E-Commerce Platform v2.0** — a big customer-facing system: React frontend, Node.js API
  with JWT auth, PostgreSQL with read replicas, Redis for sessions, Stripe payments (PCI DSS scope), a
  SageMaker recommendation endpoint. Expected load 50,000 concurrent users, 500 req/sec peak.
- **DOC-002, Internal HR Portal** — much simpler: Angular + Flask on Fargate, single MySQL instance,
  Cognito SSO against corporate AD, VPN-only access. 500 concurrent users, 20 req/sec.

Two deliberately different risk profiles, so you can watch the system reach two different decisions.

### Findings are hardcoded, and the model is told not to re-analyze

As in Modules 1 and 2, the "analysis" is a lookup, not model reasoning. `SECURITY_FINDINGS`,
`SCALABILITY_FINDINGS`, and `COST_FINDINGS` are Python dicts keyed by document ID, each holding a
pre-written `risk_level`, a `findings` list, and a `recommendation`. The tool looks up the document ID
and returns what's there. DOC-001's security entry, for instance, flags JWT tokens in `localStorage`
(XSS-vulnerable), a missing WAF on CloudFront, and unencrypted Redis — `critical_count: 2`,
`risk_level: HIGH`. DOC-002's is `critical_count: 0`, `risk_level: LOW`.

The synthesizer's system prompt makes the division of labor explicit: *"Do NOT re-analyze the
document. Base your assessment ONLY on the specialist findings."* The expensive model isn't being asked
to redo the specialists' work — only to reason across their outputs.

### Three caches, not one — and why that matters for threads

``` Python
# Shared caches — each specialist writes its findings here
security_cache = {}
scalability_cache = {}
cost_cache = {}
```

Modules 1 and 2 used a single shared dict. This lab uses **three separate dicts, one per specialist**,
each keyed by `document_id`. The change isn't cosmetic. Three threads are now writing concurrently, and
by giving each thread its own dict, no two threads ever write to the same object. Whatever one might
worry about regarding concurrent dict mutation in Python simply doesn't arise — there's nothing shared
to contend over. The synthesizer then reads all three.

> **Note:** this resolves the caveat left open in the Module 3 concept notes. The exercise does *not*
> rely on concurrent writes to one dict being safe; it structures the data so the question never comes
> up. That's the right instinct to carry forward: when threads need to report results, give each its
> own slot.

### The specialists: same builder shape, three models

Each specialist is the exact builder shape from Modules 1–2 — `BedrockModel` + system prompt + one
`@tool` — with the model matched to the task per Module 2's rule:

| Specialist | Model | Temp | Tool | Writes to |
|---|---|---|---|---|
| Security Reviewer | Nova Lite | 0.0 | `review_security` | `security_cache` |
| Scalability Reviewer | Claude | 0.1 | `review_scalability` | `scalability_cache` |
| Cost Reviewer | Nova Pro | 0.1 | `review_cost` | `cost_cache` |
| **Synthesizer** | **Claude** | **0.2** | `synthesize_reviews` | reads all three |

Every specialist system prompt demands a strict 3-line report and ends with "Do NOT add any other
commentary." The synthesizer demands exactly 5 lines. Same discipline as before: constrain the output
format so there's nothing to parse loosely.

``` Python
def build_security_agent() -> Agent:
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)
    # system_prompt: report Risk Level, Critical Issues, Recommendation in exactly 3 lines

    @tool
    def review_security(document_id: str) -> str:
        findings = SECURITY_FINDINGS.get(document_id, {})
        result = {"document_id": document_id, "domain": "security",
                  "risk_level": findings.get("risk_level"), ...}
        security_cache[document_id] = result       # this thread's own cache
        return json.dumps(result, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[review_security])
```

### The synthesizer's decision rule

The synthesizer's tool reads the three caches, takes **overall risk as the highest of the three**, and
applies a fixed decision ladder — the one place in the file that encodes policy:

``` Python
if critical_issues >= 2 or max_risk == HIGH:    decision = "BLOCK"
elif critical_issues >= 1 or max_risk == MEDIUM: decision = "APPROVE-WITH-CONDITIONS"
else:                                            decision = "APPROVE"
```

So DOC-001 (security HIGH, 2 criticals) → **BLOCK**. DOC-002 (all LOW, 0 criticals) → **APPROVE**.

### The parallel engine

``` Python
def run_specialists_parallel(doc_id: str) -> dict:
    timings = {}

    def run_security():
        return run_agent_with_retry(build_security_agent, f"Review security for document {doc_id}")
    # run_scalability and run_cost: identical one-liners

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_security): "security",
            executor.submit(run_scalability): "scalability",
            executor.submit(run_cost): "cost",
        }
        for future in as_completed(futures):
            timings[futures[future]] = future.result()
    return timings
```

Every piece is the `concurrent.futures` idiom from the primer above. `run_agent_with_retry` here
returns the **elapsed seconds** (not the agent's text, as in Module 1 — the signature changed to
`-> float`), which is what populates `timings`. After this function returns, all three caches are
populated, and only then does the orchestrator run `build_synthesizer_agent()` once. **Analysis is
parallel; synthesis is sequential** — the lesson's one-line summary, and exactly the hybrid shape the
Module 3 notes predicted.

### Demo output

| Document | Security | Scalability | Cost | Decision |
|---|---|---|---|---|
| DOC-001 | HIGH, 2 critical | MEDIUM, bottleneck at SageMaker 200 req/sec | MEDIUM, ~$12,500/mo | **BLOCK** |
| DOC-002 | LOW, 0 critical | LOW | LOW, ~$450/mo | **APPROVE** |

During the parallel phase the terminal output is interleaved and messy — three agents printing
`<thinking>` tags simultaneously. That mess is **evidence of concurrency**, not a bug: three threads
are genuinely writing to stdout at the same time. The clean summary box afterward is what to read.

### On the speedup number — and why it's smaller than it "should" be

Four different figures appear across the lesson materials: the file header says ~3×, the spoken
walkthrough says "roughly three times" and then "almost twice," and the written summary says 2–3×. None
is a constant — the demo *measures* it as `t_seq_total / t_parallel_total` each run. Two things explain
why the measured number lands nearer 2× than 3×:

1. **The synthesizer is counted in both totals.** It runs once after the parallel phase and once after
   the sequential phase, as a fixed cost added to both sides of the ratio. A fixed addend on both sides
   pulls any ratio toward 1. The *specialist phase alone* would show the ~3× the header promises; the
   full pipeline can't, by arithmetic.
2. **The narration's own worked example doesn't give 3×.** With specialists at 2s, 4s, 3s, parallel
   is 4s and sequential is 9s — that's **2.25×**, not 3×. You only approach 3× when all three
   specialists take about the same time; with uneven durations the slowest one sets the floor.

> **Note:** the honest statement is "roughly 2×, measured, with the synthesizer's fixed cost included."
> Don't quote the header's ~3× as a result of this demo — it's an upper bound on the specialist phase
> under equal durations, not what the code prints.

### Two things in the file to know about before copying it

**The decision logic is duplicated.** The BLOCK / APPROVE-WITH-CONDITIONS / APPROVE ladder appears
once inside the synthesizer's `synthesize_reviews` tool and a second time, nearly verbatim, inside
`main()`. The "Decision:" line printed in the summary box comes from **`main()`'s copy**, computed in
plain Python from the caches — *not* from the synthesizer agent's output. So the synthesizer's own
5-line report is produced but never displayed in the summary. This works, but it means two copies of
the policy that can drift apart, and it undercuts the demo's own framing that the synthesizer "makes
the final launch decision." If you build on this, keep one copy and read the decision from it.

**Every agent runs twice.** To produce the parallel-vs-sequential comparison, `main()` clears the
caches and re-runs all three specialists sequentially, then runs the synthesizer a second time too.
That's seven agent invocations per document where three-plus-one would do — four of them exist purely
to generate the timing table. Fine for a demo; a cost you would not want in anything real.

## Three things to carry forward

1. **Threads are right for LLM calls because they're I/O.** `ThreadPoolExecutor` overlaps network
   waits; it would do nothing for CPU work. Match the primitive to the bottleneck.
2. **Give each concurrent writer its own slot.** Three caches, not one — don't rely on concurrent dict
   mutation being safe when you can structure the problem so it never happens.
3. **Analysis is parallel, synthesis is sequential — and the sequential part caps your speedup.** A
   fixed synthesis cost added to both sides of the comparison pulls the ratio toward 1. Measure the
   phase you parallelized, not the whole pipeline, if you want to know what parallelism bought you.

## Module 3 wrap-up: the specialist + synthesizer architecture

The module closes by naming the pattern you built as a reusable architecture rather than a one-off.
Three skills it claims you now have, restated as the design moves they correspond to:

1. **Construct specialized agents and run them simultaneously** — aspect-based decomposition, with
   each specialist scoped to exactly one domain, executed concurrently via `ThreadPoolExecutor`.
2. **Assign independent tasks to appropriately sized models** — Module 2's model-task matching,
   applied per specialist. The module's own list of candidate domains is worth noting because it's
   wider than the lab's: *security checks, financial risk assessments, intellectual property reviews*.
   The lab did security / scalability / cost; the pattern extends to any set of independent review
   dimensions, and "intellectual property" is a reminder that the dimensions needn't be technical.
3. **Build a synthesizer that waits for the parallel phase to finish before aggregating** — the
   dependent sequential step that follows the independent parallel step. Its two defining properties:
   it *waits* (it cannot start until every specialist is done, since it reads all their outputs) and
   it *reasons across* rather than re-analyzes (its input is the specialists' findings, not the
   original document).

The reusable shape, stated once:

```
  N independent specialists  ──parallel──►  1 dependent synthesizer
  (one domain each)                          (reads all N, decides)
```

Whenever a task decomposes into independent review angles plus a final judgment that needs all of
them, this is the architecture. The capstone's Policy agent — "parallel multi-agent RAG across three
Bedrock Knowledge Bases" — is this exact shape with retrievers as the specialists.

## Key terms

- **Parallelization** — dividing a larger task into smaller, *independent* sub-tasks that run
  concurrently to reduce overall execution time. The independence is load-bearing; it's what makes
  concurrent execution safe.
- **Specialist Agent** — an LLM-powered agent scoped and configured to evaluate exactly one domain or
  function. The "single-responsibility agent" of Module 1, now named for its role in a parallel
  fan-out.
- **Synthesizer Agent** — a final-stage agent that aggregates, reasons across, and summarizes the
  outputs of multiple upstream specialists. It is the dependent step: it cannot begin until the
  specialists finish, and it works from their findings, not from the original input.
- **ThreadPoolExecutor** — Python's execution engine (in `concurrent.futures`) that manages a pool of
  worker threads, enabling concurrent execution of submitted tasks. The right primitive for I/O-bound
  work like LLM API calls, because threads overlap network waits.

# Module 4: Agent Orchestration

## Three patterns in one workflow

Module 1 used one pattern (sequential). Module 3 used one pattern (parallel). This module's claim is
that real workflows need **all three at once**, chosen per phase by the shape of the dependencies:

| Pattern | Use it when | The tell |
|---|---|---|
| **Sequential** | steps depend on each other | B needs A's output |
| **Parallel** | steps are independent | none needs another's output |
| **Conditional** | the path depends on the data | an `if` on a field decides which agent runs |

The scenario is employee onboarding: create an account, assign a manager, provision a laptop, set up
email, grant building access, then department-specific setup. Some steps depend on each other (you
can't assign a manager to an account that doesn't exist), some are independent (laptop and email
don't care about each other), and some depend on who the person is (engineers get GitHub and
PagerDuty, salespeople get Salesforce and Gong). One workflow, three phases, three patterns.

```
            New Employee
                 │
   ┌─────────────▼─────────────┐
   │ PHASE 1 — SEQUENTIAL      │  AccountCreator → ManagerAssigner
   │ (manager needs account)   │  B waits for A
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ PHASE 2 — PARALLEL        │  Laptop ‖ Email ‖ Building
   │ (ThreadPoolExecutor)      │  all three at once; phase time = slowest
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ PHASE 3 — CONDITIONAL     │  if Engineering → EngineeringOnboarding
   │ (plain if/elif on data)   │  if Sales       → SalesOnboarding
   └─────────────┬─────────────┘
                 │
          Onboarding Complete
```

## The orchestrator is Python, not an LLM — and why that is the whole point

The orchestrator `orchestrate_onboarding` is a plain Python function. It runs Phase 1 as two
sequential calls, fans Phase 2 out to a `ThreadPoolExecutor`, and picks Phase 3's agent with an
`if/elif` on `emp["department"]`. No model makes any of those decisions.

The module is emphatic about this, and the reason is **determinism**: same input always produces the
same execution order. That buys three things an LLM orchestrator cannot give you: you can *debug* it
(trace exactly which step failed), *test* it (unit-test each phase), and *trust* it (no risk of a model
forgetting a step or reordering phases). This is Module 1's "the coordinator is not an agent" rule,
now extended to cover parallel fan-out and conditional branching as well — the orchestrator's job grew,
and it still did not need a model to do it.

The division of labor is sharp: **workers are LLM agents, the orchestrator is code.** The intelligence
lives in the six workers; the control flow lives in Python. Neither does the other's job.

## Worked example: HR onboarding

Demo file: `lab_4/Demo/hr_onboarding.py`. Three employees, each chosen to exercise one part of the
workflow:

| Employee | Department | Purpose |
|---|---|---|
| EMP-001 Alice Chen | Engineering | the engineering path through Phase 3 |
| EMP-002 Bob Martinez | Sales | the sales path — same workflow, different Phase 3 outcome |
| EMP-003 Carol Johnson | Engineering (DevOps) | carries `simulate_failure: True` — meant to demonstrate retry |

### All six workers share one shape, and all use Nova Lite

Every worker is the Module 1–2 builder: a `BedrockModel` at `temperature=0.0`, a one-line system
prompt, one `@tool`. **All six use Nova Lite**, per Module 2's rule — these are straightforward lookup
tasks, none needs a stronger model. The tools write into a shared `workflow_state` dict, each under its
own key (`account`, `manager`, `laptop`, `email`, `building`, `department_onboarding`).

``` Python
def build_account_creator() -> Agent:
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)
    system_prompt = """You are an account creation agent. Your ONLY job:
1. Call create_account with the employee_id
2. Report: Account created for <name> (ID: <employee_id>)
Do NOT add any other commentary."""

    @tool
    def create_account(employee_id: str) -> str:
        # ... look up employee, build account record ...
        workflow_state["account"] = result        # this worker's own key
        return json.dumps(result, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[create_account])
```

> **Note:** the file's header comment says "Claude 3 Sonnet for orchestrator" and it imports a
> `CLAUDE_MODEL` constant — but **`CLAUDE_MODEL` is never used anywhere in the file**. There is no
> orchestrator model, because the orchestrator is plain Python, exactly as the lesson says. The
> header comment contradicts the code; the code is right.

### One shared dict across threads — the opposite of Module 3's choice, and why it still works

Module 3's lab used three separate caches, one per parallel worker. This demo uses a **single**
`workflow_state` dict that three Phase 2 threads write to concurrently. That looks like the thing
Module 3 avoided — but it's safe for the same underlying reason: **each tool writes a distinct key.**
`laptop`, `email`, and `building` never collide. Two threads writing *different* keys of one dict is
as safe as two threads writing different dicts; what you must never do is have two threads write the
*same* key. The exercise's "Common Mistakes" section states this rule explicitly (see below). Both
designs honor it; they just draw the boundary in different places.

### The orchestrator

``` Python
def orchestrate_onboarding(employee_id: str) -> dict:
    emp = next(e for e in EMPLOYEES if e["id"] == employee_id)

    # PHASE 1 (SEQUENTIAL): account must exist before manager assignment
    run_agent_with_retry(build_account_creator,  f"Create account for {employee_id}")
    run_agent_with_retry(build_manager_assigner, f"Assign manager for {employee_id}")

    # PHASE 2 (PARALLEL): laptop, email, building access are independent
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_laptop):   "laptop",
            executor.submit(run_email):    "email",
            executor.submit(run_building): "building",
        }
        for future in as_completed(futures):
            timings[futures[future]] = future.result()

    # PHASE 3 (CONDITIONAL): a code decision, not an LLM decision
    if emp["department"] == "Engineering":
        run_agent_with_retry(build_engineering_onboarding, f"Onboard engineer {employee_id}")
    elif emp["department"] == "Sales":
        run_agent_with_retry(build_sales_onboarding,       f"Onboard sales {employee_id}")
```

Phase 2 is Module 3's `ThreadPoolExecutor` + `as_completed` + dict-of-futures idiom, unchanged. Phase
3 is an `if/elif` — the file also has an `else` that prints "Unknown department" and skips, which is
the right thing for a value the branches don't cover.

### The retry demo does not demonstrate retry — a real defect, worth understanding

The narration promises that Carol's laptop provisioning "fails on the very first attempt," that you
"see a retry message," and that it "succeeds on the second attempt" via exponential backoff. **The code
cannot produce that.** Here is why, traced through:

`run_agent_with_retry` retries only on an **exception**:

``` Python
try:
    agent = agent_builder()
    agent(prompt)
    return elapsed
except Exception as e:          # ← retry fires ONLY here
    ... sleep(2 ** attempt) ...
```

But the simulated failure lives *inside the tool*, and it **returns** rather than raises:

``` Python
if emp.get("simulate_failure") and "laptop_attempts" not in workflow_state:
    workflow_state["laptop_attempts"] = 1
    return json.dumps({"error": "Provisioning system temporarily unavailable", "retry": True})
```

A returned error string is a *normal tool result*. The agent receives it, reports it in prose, and the
call completes without exception. `run_agent_with_retry` sees success and returns. No `[Retry 1/3]`
line prints, no backoff happens, and — critically — `workflow_state["laptop"]` is **never written** for
Carol, because the only code path that writes it is the one the simulated failure skipped. Her summary
box will show `Laptop: ? (?)`.

What `run_agent_with_retry` *does* protect against is a real Bedrock exception — throttling, a network
error — which the demo has no way to trigger on demand. So the mechanism is sound; the demonstration of
it is not. The fix would be either to `raise` inside the tool (so the exception propagates up through
the agent to the retry wrapper), or to have the orchestrator inspect `workflow_state` after the call
and re-invoke on an error result. As written, the "failure handling" phase of this demo is a claim the
output does not back up.

> **Note:** this is worth remembering as a general rule, not just a bug report. **A tool that returns
> an error has not failed — it has succeeded at returning an error.** The agent will happily report it.
> If you want a failure to propagate to a retry mechanism, the tool has to raise, or something outside
> the agent has to check the result. "Return an error dict" and "trigger a retry" are different things.

### Demo output

| Employee | Phase 3 route | Tools provisioned |
|---|---|---|
| EMP-001 Alice | Engineering | GitHub Enterprise, Jira, AWS Console, PagerDuty |
| EMP-002 Bob | Sales | Salesforce, HubSpot, Gong, LinkedIn Sales Navigator |
| EMP-003 Carol | Engineering | same as Alice — but laptop shows `?` per the defect above |

Phase 2's reported total is the **slowest** of the three parallel workers, not the sum — Module 3's
principle, visible again. Phases 1 and 3 are sequential and add.

## The exercise: package delivery, with a gate

Exercise file: `lab_4/delivery_workflow.py` (at the `lab_4/` root, not under `Demo/`). Same three-phase
shape as the demo, same six-worker count, same single `workflow_state` dict — with one twist in each
phase.

```
         New Delivery Request
                 │
   ┌─────────────▼─────────────┐
   │ PHASE 1 — SEQUENTIAL GATE │  AddressValidator → valid?
   │                           │     NO  → HALT (return early; nothing else runs)
   │                           │     YES → continue
   └─────────────┬─────────────┘
                 │ only if valid
   ┌─────────────▼─────────────┐
   │ PHASE 2 — PARALLEL        │  Label ‖ Insurance ‖ Carrier
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ PHASE 3 — CONDITIONAL     │  sender_country == country → Domestic
   │                           │  else                      → International (+ customs)
   └─────────────┬─────────────┘
                 │
          Delivery Processed
```

### Phase 1 — a gate, not a chain

This is the new pattern. The demo's Phase 1 was a **chain**: A then B, both always run. The exercise's
Phase 1 is a **gate**: run one validator, and if it fails, **stop the entire workflow**. Still
sequential, but instead of "A then B" it's "validate or halt."

``` Python
run_agent_with_retry(build_address_validator, f"Validate address for {package_id}")
validation = workflow_state.get("validation", {})
if validation.get("status") != "valid":
    return {"halted": True, "halt_reason": validation.get("reason"), ...}   # ← the gate
```

The `return` is the entire mechanism. It exits `orchestrate_delivery` before Phase 2 or Phase 3 can
run. For PKG-003 (empty address), that means no label, no insurance, no carrier selection, no
shipping — zero Bedrock calls beyond the one validator. The gate's value is **not spending compute on
work the precondition has already ruled out.**

The validator itself checks two things in the tool: is the address empty, or shorter than 10
characters? Either → `status: "invalid"` with a reason, written to `workflow_state["validation"]` for
the orchestrator to inspect.

### Phase 3 — a different routing key

The demo routed on `department == "Engineering"`. The exercise routes on whether two fields match:

``` Python
is_domestic = pkg["sender_country"] == pkg["country"]
if is_domestic:   run_agent_with_retry(build_domestic_shipping, ...)
else:             run_agent_with_retry(build_international_shipping, ...)
```

The international branch is where the only real logic difference lives — inside the tool body:

``` Python
customs_duty = round(pkg["declared_value"] * 0.05, 2)                       # 5% duty
total = round(carrier["rate"] + insurance["premium"] + customs_duty, 2)
# plus a customs_declaration block with hs_code "8543.70", customs_required=True
```

Domestic sums carrier + insurance only, sets `customs_required=False`, no declaration.

### The three packages

| Package | Route | What it tests |
|---|---|---|
| PKG-001 | US → US, $150 electronics | the straight-through path: gate passes, basic insurance, cheapest US carrier (USPS Priority $12.50), domestic |
| PKG-002 | US → DE, $2,500 industrial | the international path: premium insurance tier (value > $1,000), DHL International, **customs duty = 5% × $2,500 = $125** |
| PKG-003 | US → US, **empty address** | **the gate**: validator fails, workflow halts, Phase 2 and 3 show `0` in the summary table |

The insurance tier is a threshold ladder in the tool (`> 1000` → premium at 5%, `> 500` → standard at
3.5%, else basic at 2%), and the carrier selector is `min(carriers, key=rate)` — both plain code, both
in the tool, neither an LLM decision.

### Where the lesson text understates the file

The lesson text's Phase 2 snippet for the exercise uses a **list** of futures with no labels
(`futures = [executor.submit(...), ...]`) and just `f.result()`s each. The actual file uses the
labeled dict-of-futures idiom and records per-worker timings, same as the demo and Module 3. The text
simplified; the file is the better pattern.

## Common mistakes the module names — all three verifiable in the files

**Missing the workflow halt.** A gate needs a hard stop. Without the explicit `return` after a failed
validation, Phase 2 and 3 run anyway and you've spent compute on a package with no address. The
`return` statement *is* the gate; everything else is just a validator that gets ignored.

**Using LLMs for deterministic logic.** It's tempting to ask a routing agent "is this domestic or
international?" But `is_domestic = sender_country == dest_country` is faster, cheaper, and cannot be
wrong. Every routing decision in both files is code. Reserve models for natural-language work and
genuine reasoning; a field comparison is neither.

**Global state race conditions.** When threads share a dict, each must write to **distinct,
non-overlapping keys**. Both files honor this — `laptop`/`email`/`building` in the demo,
`label`/`insurance`/`carrier` in the exercise. This is the same rule Module 3 solved with separate
dicts; here it's solved with separate keys in one dict. The rule is about *what is written*, not *how
many dicts there are*.

## Four things to carry forward

1. **Pick the pattern per phase by the dependency shape.** Dependent → sequential. Independent →
   parallel. Data-dependent path → conditional. One workflow routinely needs all three.
2. **The orchestrator stays code even as it gets more capable.** Fan-out, branching, gating, retry —
   none of it needed a model. Determinism is what makes it debuggable, testable, and trustworthy.
3. **A gate is a sequential precondition with an early `return`.** It's the only pattern here that can
   *stop* the workflow, and its whole value is the compute it refuses to spend.
4. **A tool that returns an error has not failed.** Retry mechanisms catch exceptions; a returned error
   dict is a successful call. If you want retry, raise — or check the result outside the agent. The demo
   gets this wrong, and the lesson doesn't notice.

## Key terms

- **Orchestration** — the automated configuration, coordination, and management of multiple agents
  and system tasks. Here, done by a plain Python function.
- **Deterministic Workflow** — a process whose sequence and routing are governed by code rules, not
  by probabilistic model output. Same input → same execution order, always.
- **Sequential Execution** — steps that depend on the previous step's completion. Two flavors seen
  here: the **chain** (A then B, both always run) and the **gate** (validate or halt).
- **Sequential Gate** — a precondition that must pass before any subsequent step executes; on failure
  the workflow halts via early return.
- **Parallel Execution** — independent tasks run simultaneously to cut total time; phase time is the
  slowest task, not the sum.
- **Conditional Routing** — a deterministic code decision that evaluates data and directs the workflow
  down one of several paths. Explicitly *not* an LLM decision.
- **The four building blocks**, as the exercise names them: **chains, gates, fan-out, routing.**

# Module 5: Routing and Data Flow

## When the orchestrator's plan isn't enough

Modules 1–4 all shared one property: **every request went through the same set of agents**, in an
order the orchestrator already knew. Module 4's conditional routing chose *which* Phase 3 agent ran,
but the decision was a field comparison the orchestrator could make instantly. This module is for the
case where the orchestrator *doesn't* know the plan up front — a request arrives, and the first job is
figuring out which specialist should handle it at all.

## Three routing patterns, in the abstract

| Pattern | What it inspects | Use it for |
|---|---|---|
| **Content-based routing** | the message itself | sorting diverse incoming requests to the right specialist |
| **Round-robin routing** | nothing about the message — just whose turn it is | spreading load evenly across *identical* agents, for scaling |
| **Priority-based routing** | some urgency signal on the message | treating urgent work differently from routine work |

Content-based is the one this module builds. Round-robin and priority are named but not implemented
here — round-robin because it's a load-balancing concern rather than a correctness one, priority
because it shows up as Tier 1 of the hybrid router below rather than as a standalone pattern.

## Data flow management, in the abstract

Routing decides *where* a message goes. Data flow management handles the fact that the payload may not
be in the shape the receiving agent needs. Three operations, named once:

- **Enhancement** — adding related information before forwarding.
- **Filtering** — removing data the receiver doesn't need.
- **Reformatting / transformation** — changing structure or types.

The demo barely exercises this (the `input_text[:80]` truncation before logging is a filter, loosely),
so these are vocabulary to hold onto rather than something the lab works through.

## The hybrid router: four tiers, cheapest first

Demo file: `lab_5/Demo/financial_router.py`. The scenario is a financial transaction system where
requests arrive as free text — wire transfers, fraud reports, balance inquiries, high-value
transactions — and each type needs a different specialist. The router is a **four-tier decision
pipeline**; a request flows through the tiers in order and **the first match wins**.

```
          Incoming Request
                │
      ┌─────────▼─────────┐
      │ TIER 1 — PRIORITY │  amount > $10,000?  → SeniorReviewAgent, done.
      │  (one comparison) │  else fall through
      └─────────┬─────────┘
                │
      ┌─────────▼─────────┐
      │ TIER 2 — RULES    │  regex on the text: wire/transfer → Payments
      │  (regex, free)    │                     fraud/stolen  → Fraud
      │                   │                     balance/stmt  → Account
      └─────────┬─────────┘  no match → fall through
                │
      ┌─────────▼─────────┐
      │ TIER 3 — LLM      │  Nova Lite classifier → {intent, confidence}
      │  (costs a call)   │  confidence ≥ 0.6 → route to that intent's agent
      └─────────┬─────────┘  below 0.6 → fall through
                │
      ┌─────────▼─────────┐
      │ TIER 4 — FALLBACK │  GeneralSupportAgent, flagged for human review.
      │  (safety net)     │  Nothing is ever dropped.
      └───────────────────┘
```

**The ordering is the whole design.** Priority first, then rules, then LLM, then fallback — always the
cheapest, fastest option first. A number comparison runs in nanoseconds. A regex runs in microseconds.
An LLM call runs in seconds and costs money. By trying them in that order, the expensive tier only ever
sees the requests the cheap tiers couldn't handle.

### The dispatcher

``` Python
ROUTING_RULES = [
    (r"\b(wire|transfer|send money|payment)\b",            "PaymentsAgent"),
    (r"\b(fraud\w*|stolen|unauthorized|suspicious)\b",     "FraudAgent"),
    (r"\b(balance|statement|account info|account history)\b", "AccountAgent"),
]
CONFIDENCE_THRESHOLD = 0.6

def hybrid_route(request: dict) -> dict:
    text = request["text"]

    # 1. Priority — business-critical override
    if request.get("amount", 0) > 10000:
        return {"target_agent": "SeniorReviewAgent", "method": "priority", "confidence": 1.0}

    # 2. Rules — fast, free, deterministic
    for pattern, agent_name in ROUTING_RULES:
        if re.search(pattern, text.lower()):
            return {"target_agent": agent_name, "method": "rule", "confidence": 1.0}

    # 3. LLM classification — flexible, handles ambiguity
    agent_name, confidence, _ = llm_classify(text)
    if confidence >= CONFIDENCE_THRESHOLD:
        return {"target_agent": agent_name, "method": "llm", "confidence": confidence}

    # 4. Fallback — flag for human review
    return {"target_agent": "GeneralSupportAgent", "method": "fallback", "confidence": confidence}
```

Four `return`s, one per tier. Tiers 1 and 2 report `confidence: 1.0` because they are deterministic —
a rule either matched or it didn't. Only Tier 3 produces a real confidence number.

### Why priority must come before rules — not just for cost

The lesson frames the ordering as a cost optimization. It is also a **correctness requirement**, and
the test data proves it. TXN-007 reads *"Wire $50,000 to an overseas account"* and TXN-008 reads *"I
need to transfer $25,000"*. Both contain `wire` / `transfer` — both **would match the Payments rule**.
They route to SeniorReview only because the priority check fires first and returns before the rules
ever run. Swap Tiers 1 and 2 and both high-value transactions would misroute to PaymentsAgent with
`confidence: 1.0`. The override only overrides because it is evaluated first.

> **Note:** this is the general lesson about tiered routers. The tiers are not independent filters
> that happen to be ordered by cost; a later tier can *also* match a request an earlier tier claimed.
> The order encodes precedence, and precedence is policy.

### The classifier: the model supplies its own confidence

When Tier 3 runs, it calls a Nova Lite classifier whose system prompt locks the response to a single
`classify_intent` tool call returning `{intent, confidence}`:

``` Python
@tool
def classify_intent(intent: str, confidence: float) -> str:
    classification_result["intent"] = intent.lower().strip()
    classification_result["confidence"] = min(max(float(confidence), 0.0), 1.0)   # clamp to [0, 1]
    return json.dumps(classification_result)
```

Read the arguments carefully: **the LLM passes `confidence` in as a tool argument.** The tool does
not compute confidence from anything — it records the model's self-report and clamps it to `[0, 1]`
in case the model produces something out of range. This is the first place in the course where a
model's own judgment, rather than a dictionary lookup, feeds a routing decision. The 0.6 threshold is
the guard against trusting that self-report blindly: below it, the request goes to a human instead.

Forcing the output through a tool call (rather than free text) is the same "read structured fields,
never parse prose" discipline from Module 1, applied to classification.

### The test set, and which tier each request hits

| Requests | Text pattern | Amount | Tier | Agent |
|---|---|---|---|---|
| TXN-001, 002 | "wire…", "transfer…" | $5,000 / $2,000 | rule | PaymentsAgent |
| TXN-003, 004 | "stolen… unauthorized", "fraudulent" | — | rule | FraudAgent |
| TXN-005, 006 | "balance", "statement" | — | rule | AccountAgent |
| TXN-007, 008 | "Wire $50,000…", "transfer $25,000…" | **$50,000 / $25,000** | **priority** | SeniorReviewAgent |
| TXN-009 | "help with my money situation, things are complicated" | — | llm | whatever the model says |
| TXN-010 | "purple elephant moonlight dancing xkcd random words" | — | fallback | GeneralSupportAgent |

Method distribution: **2 priority, 6 rule, 1 LLM, 1 fallback.** Eight of ten requests — 80% — never
touched the LLM. That is the cost story in one number.

> **Note:** the file's own header comment says rules handle "70%" and the LLM "30%"; the narration
> says rules handle "maybe 60%"; the test set gives 60% rule + 20% priority = 80% non-LLM. These are
> all illustrative, not measured on real traffic. The point they share is that most requests should be
> cheap, and the exact split depends entirely on how predictable your traffic is.

### "10 out of 10 correct" — read the fine print

The correctness check in `main()` is:

``` Python
correct = (expected in ("LLM-classified",) or target == expected)
```

TXN-009's `expected_agent` is the literal string `"LLM-classified"`. So **TXN-009 is marked correct no
matter what the classifier returns** — if it classified "my money situation" as fraud, or fell through
to fallback, it still counts as a match. The accuracy claim is really nine genuinely-checked requests
plus one that cannot fail. That's a reasonable choice for a demo whose point is the *tier* rather than
the *answer*, but "10/10" overstates what was verified.

### The workers: five specialists, all Nova Lite

| Agent | Tool | What it does |
|---|---|---|
| PaymentsAgent | `process_payment` | marks a payment processed, issues a `PAY-` reference |
| FraudAgent | `investigate_fraud` | opens a case; `risk_level` is `high` if amount > $1,000 else `medium` |
| AccountAgent | `lookup_account` | returns a fixed balance and statement date |
| SeniorReviewAgent | `senior_review` | `risk_assessment` is `elevated` if amount > $25,000 else `moderate`; sets `approval_required: True` |
| GeneralSupportAgent | `general_support` | opens a ticket with `status: awaiting_human_review` |

All five use Nova Lite, per Module 2's rule — and the file's own justification is worth quoting: *"the
routing decision has already been made. The workers just execute."* The hard thinking happened in the
router; the workers are lookups. A dict `AGENT_BUILDERS` maps agent names to builder functions, so
dispatch after routing is one line: `builder = AGENT_BUILDERS[target]`.

## Audit logging to DynamoDB — the first real AWS resource in the course

Every routing decision is written to a DynamoDB table. This is the first module whose demo depends on
infrastructure you have to provision; everything before ran against Bedrock alone.

### The table

Created by `lab_5/Demo/stack.yaml`, a CloudFormation template with one resource:

``` YAML
RoutingAuditTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: !Sub ${ProjectName}-routing-audit      # default: lesson-05-demo-routing-audit
    BillingMode: PAY_PER_REQUEST                      # on-demand — no capacity to size
    KeySchema:
      - AttributeName: request_id   KeyType: HASH     # partition key
      - AttributeName: timestamp    KeyType: RANGE    # sort key
    TimeToLiveSpecification:
      AttributeName: ttl
      Enabled: true
```

Three design choices worth noticing:

- **`PAY_PER_REQUEST`** — on-demand billing. Nothing to provision, nothing idle to pay for. Right for a
  demo and for any audit table whose write volume is unpredictable.
- **Composite key: `request_id` + `timestamp`.** The partition key alone wouldn't work, because the
  same request ID could in principle be logged more than once; the sort key makes each (request, moment)
  pair unique and lets you query one request's history in time order.
- **TTL on an attribute named `ttl`.** DynamoDB will delete items once their `ttl` epoch-seconds value
  is in the past. The table is self-cleaning; the code decides the retention.

### The log entry

``` Python
def log_routing_decision(request_id, input_text, method, target_agent, confidence, latency_ms):
    entry = {
        "request_id":     request_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "input_text":     input_text[:80],                     # truncated — a filter
        "routing_method": method,
        "target_agent":   target_agent,
        "confidence":     str(round(confidence, 2)),
        "latency_ms":     str(round(latency_ms, 1)),
        "ttl":            int(time.time()) + 86400,            # auto-delete after 24h
    }
    audit_table.put_item(Item=entry)
```

Each entry answers the compliance questions: *which agent got this, decided by which method, how
confident, how fast.* The 24-hour TTL is set **per item, in code** — the table's TTL setting only says
*which attribute* to honor; the application decides the value. Change `86400` and retention changes
with no infrastructure touch.

> **Note:** `confidence` and `latency_ms` are stored as **strings**, not numbers. That sidesteps a
> real DynamoDB annoyance — boto3 rejects Python `float` and requires `Decimal` for numeric attributes —
> but it means you can't do numeric queries or comparisons on those fields later. A conscious trade for
> a demo; in a real audit table you'd convert to `Decimal` and keep them numeric.

### Two things the demo does that you shouldn't copy

**It reads the audit log back with `scan()`.** At the end, `audit_table.scan()` pulls every item to
print the table. Fine for ten rows. `scan` reads the *entire* table, costs read capacity proportional
to table size, and is the wrong tool for anything beyond a demo — use `query` on a partition key, or a
GSI, for real lookups.

**It fails hard if the table is missing.** `_verify_audit_table()` runs first and does `SystemExit(1)`
if the table can't be reached, printing the `aws cloudformation deploy` command to fix it. That is the
*right* behavior — an audit log that silently drops writes is worse than one that refuses to start —
but it means the demo has a hard infrastructure prerequisite, unlike every prior module.

## Four things to carry forward

1. **Try the cheapest routing method first, and only escalate.** Comparison → regex → LLM → fallback.
   The expensive tier should only ever see what the cheap tiers couldn't handle.
2. **Tier order is precedence, not just cost.** A later tier can match a request an earlier tier
   claimed. Put overrides first because they must *win*, not only because they're fast.
3. **A model's confidence is a self-report.** The classifier hands its own confidence to the tool as
   an argument. Clamp it, threshold it, and send low-confidence cases to a human — don't trust it raw.
4. **Log every automated decision somewhere you can defend it later.** Agent, method, confidence,
   latency, timestamp — with a TTL so the table cleans itself. If a routing decision is ever
   questioned, the audit row is the answer.

## Key terms

- **Hybrid routing** — combining multiple evaluation strategies (deterministic rules, priority checks,
  LLM classification) in a fixed order to optimize speed, cost, and accuracy together.
- **Content-based routing** — inspecting the message itself to decide its destination.
- **Priority-based routing** — treating urgent requests differently; here, the first tier.
- **Round-robin routing** — spreading load evenly across identical agents; named, not built.
- **Fallback strategy** — the default path when every primary method fails or returns low confidence,
  so no request is ever dropped.
- **Audit logging** — recording automated decisions to a durable store for compliance and debugging.
- **Data flow management** — transforming payloads between agents: enhancement, filtering,
  reformatting.

# Module 6: State Management

## What "state" means here, and why it's where projects stumble

Every module so far shared state through a Python dict that lived for one process run and died with
it. That was enough because no module needed to remember anything across runs, and no two agents ever
wrote the *same* key. This module is about the two things that dict can't do: **persist** across
interruptions and sessions, and **survive concurrent writes to the same record** without losing data.

The module names this as the place multi-agent projects most often fail, and the diagnosis is simple:
agents almost never operate in isolation. They need context gathered earlier. If the orchestrator
doesn't manage that context deliberately, the quoting agent can't apply a discount because it doesn't
know the customer, and the fulfillment agent can't ship because it doesn't have the address. **Agents
without managed state are flying blind.**

Two kinds of state to distinguish from the outset:

| Kind | What it is | Lifetime | Example from this module |
|---|---|---|---|
| **Conversation-level** (within-session) | what's happening right now in this workflow | one request / one transaction | the trip record: driver, fare, ETA, version |
| **System-level** (cross-session) | what should be remembered for next time | across sessions, across process restarts | a rider's preferred driver |

The module's one-sentence architectural claim: these need **two different services**. Within-session
transactional state goes in DynamoDB. Cross-session conversational context goes in AgentCore Memory.
They have different consistency requirements, different lifetimes, and different access patterns, and
forcing one store to do both jobs is how you get either a slow memory or an inconsistent transaction.

## Stateful orchestration: pass only what each agent needs

The order-processing worked example makes the discipline concrete. For a request like *"Quote 10 reams
of PROD_A for ACME_CORP, needed by next Friday,"* the orchestrator holds the full state:

```
customer_id       = ACME_CORP          request_timestamp = 2025-04-30
order_id          = ORD-12345          needed_by_date    = 2025-05-09
order_items       = [{product_id: PROD_A, quantity: 10}]
(later)             inventory_status, quote_details, shipping_address, payment_status …
```

But when it delegates, **it does not dump the whole state on each worker.** It passes only what that
task needs:

| Worker | Gets | Does *not* get |
|---|---|---|
| Inventory agent | `product_id`, maybe `quantity` | `customer_id`, `needed_by_date` — a stock check doesn't need them |
| Quoting agent | `order_items`, `customer_id` (for discounts), maybe `inventory_status` | shipping details |
| Fulfillment agent | `order_id`, `order_items`, `shipping_address`, `needed_by_date` | pricing internals |

This is the same single-responsibility instinct from Module 1, applied to *data* instead of *tasks*.
A worker that receives only its slice can't accidentally act on fields that aren't its business, and
its prompt stays small. The module lists four mechanisms for the handoff — direct function arguments,
a message object, a shared state dictionary managed by the framework (graph-based systems), or
straight into a tool call — and every lab so far has used the third.

## Handling failure: five strategies, one level up from tool errors

Module 4 handled failure at the tool level (`run_agent_with_retry`). This is the next layer: what the
*orchestrator* does when a step fails. Five strategies, roughly in order of escalation:

1. **Retry logic** — for transient issues: a network blip, a rate limit. Try again, usually with
   exponential backoff (1s, 2s, 4s). Already seen in every lab since Module 1.
2. **Fallbacks / alternative paths** — if retries fail or the error is clearly permanent, execute
   Plan B: route to a different agent, use a simpler backup tool, branch the workflow. Module 5's
   fallback tier is exactly this.
3. **Graceful failure reporting** — if the workflow truly cannot proceed, don't crash silently.
   Terminate cleanly and report back with clear context. Module 4's gate halt is this.
4. **Compensating actions** — the new one, and the one that matters most when a workflow mutates
   external state. If Step 1 *charge credit card* succeeds and Step 2 *provision service* fails
   irrecoverably, you must **undo Step 1**. Otherwise you've charged a customer for nothing. Any
   workflow that modifies external state needs to think about atomicity and compensation up front.
5. **Human-in-the-loop escalation** — for errors too complex, ambiguous, or critical to automate.
   Pause, create a high-priority ticket with the inputs, intermediate results, and error logs
   attached, and let a person decide.

> **Note:** compensating actions are the strategy this module names but does not implement. The
> ride-sharing demo never has to undo anything — if the pricing agent fails, no money has moved. Keep
> it in mind for the capstone's Refund agent, which is exactly the kind of step that may need undoing.

## The demo: three agents, one DynamoDB record

Demo file: `lab_6/Demo/ride_sharing_state.py`. Infrastructure: `lab_6/Demo/stack.yaml`. The scenario
is a ride-sharing system where a trip request needs three things done — match a driver, calculate the
fare, estimate arrival — and **all three agents write their result into the same trip record.**

```
                  Rider requests trip
                          │
        ┌─────────────────▼─────────────────────┐
        │  DynamoDB: lesson-06-shared-state-     │
        │            trip-state                  │
        │  trip_id (PK) │ version │ driver │     │
        │  fare │ eta │ status │ ttl             │
        └──────┬──────────┬──────────┬───────────┘
               │          │          │
        ┌──────┴───┐ ┌────┴────┐ ┌───┴─────┐
        │ Driver   │ │ Pricing │ │  ETA    │   three agents, ONE record
        │ Match    │ │         │ │         │   each writes its own field
        └──────────┘ └─────────┘ └─────────┘
               │
        ┌──────▼──────────────────────┐
        │ rider_memory (in-process)   │   "AgentCore Memory simulation"
        │ preferred_driver per rider  │
        └─────────────────────────────┘
```

### The table

One resource in `stack.yaml`, simpler than Module 5's:

``` YAML
TripStateTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: !Sub ${ProjectName}-trip-state       # default: lesson-06-shared-state-trip-state
    BillingMode: PAY_PER_REQUEST
    KeySchema:
      - AttributeName: trip_id   KeyType: HASH      # partition key ONLY — no sort key
    TimeToLiveSpecification:
      AttributeName: ttl
      Enabled: true
```

Module 5's audit table had a composite key (`request_id` + `timestamp`) because the same request could
be logged many times. This table has **only a partition key**, because there is exactly one record per
trip and every agent updates *that one record in place*. That's the whole point — a single row that
several writers contend over. The `version` attribute is not part of the key; it's an ordinary field
the application uses as a lock token. TTL is set to one hour at creation (`now + 3600`), so completed
trips clean themselves up.

## The problem: lost updates

What happens when two agents update the same record at the same time? The naive sequence:

```
  time →
  Pricing agent:   READ record (driver=null, fare=null)  →  ... compute ...  →  WRITE {fare: 37.75}
  ETA agent:          READ record (driver=null, fare=null)  →  ... compute ...  →  WRITE {eta: 29}
```

If both do a full-record `put_item`, whoever writes **last wins** and silently erases the other's
field. The ETA agent read a record with `fare=null`, so when it writes the whole record back, `fare` is
null again — the pricing agent's work is gone, and nothing raised an error. That is a **lost update**,
and it is silent, which is what makes it dangerous.

## The solution: optimistic locking — `update_trip` walked through

"Optimistic" because it assumes conflicts are rare and doesn't lock anything up front. Instead, each
record carries a **version number**, and every write says: *only save this if the version is still what
I read.* If someone else wrote first and bumped the version, the write is rejected and you try again.

Here is `update_trip`, the function every agent calls, annotated line by line:

``` Python
def update_trip(trip_id: str, updates: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):

        # ── STEP 1: READ ─────────────────────────────────────────────────────
        # Fetch the record and capture its version. This number is the "lock
        # token": we must hand it back unchanged when we write, as proof that
        # nobody else has written in between.
        response = trip_table.get_item(Key={"trip_id": trip_id})
        current = response.get("Item")
        if not current:
            raise KeyError(f"Trip {trip_id} not found")
        expected_version = int(current["version"])          # e.g. 2

        # ── STEP 2: MODIFY (locally — nothing has hit DynamoDB yet) ──────────
        # Merge in this agent's updates and bump the version by one.
        current.update(to_dynamo(updates))                  # e.g. add fare = {...}
        current["version"] = expected_version + 1           # 2 → 3
        current["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # ── STEP 3: CONDITIONAL WRITE ────────────────────────────────────
            # Put the whole record back, BUT only if the stored version still
            # equals what we read in step 1. DynamoDB evaluates this check
            # atomically on the server — no gap between "check" and "write".
            trip_table.put_item(
                Item=current,
                ConditionExpression="version = :expected_ver",
                ExpressionAttributeValues={":expected_ver": expected_version},
            )
            return from_dynamo(current)                     # success — we're done

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # ── CONFLICT ─────────────────────────────────────────────────
                # Someone else wrote between our READ and our WRITE, so the
                # stored version is no longer expected_version. Our local copy
                # is stale. Wait briefly, then loop back to STEP 1 and re-read.
                if attempt < max_retries - 1:
                    wait = 0.1 * (2 ** attempt)             # 0.1s, 0.2s, 0.4s
                    time.sleep(wait)
                else:
                    raise VersionConflictError("Version conflict after retries")
            else:
                raise                                       # any other error: real failure
```

**Walking a conflict through.** Suppose the record is at version 2. Pricing and ETA both read it and
both capture `expected_version = 2`. Pricing writes first with condition `version = 2` — the stored
version is 2, so the write succeeds and the stored version becomes 3. Now ETA writes with condition
`version = 2` — but the stored version is 3. **DynamoDB rejects it** with
`ConditionalCheckFailedException`. ETA sleeps 0.1s, loops to STEP 1, re-reads, sees version 3 *and*
sees Pricing's `fare` field now present in the record, applies its own `eta` update on top of that
fresh copy, writes with condition `version = 3`, succeeds. Both fields survive. Nothing was lost.

**Three things about this that are easy to miss:**

- **The check is atomic on the server.** DynamoDB compares and writes in one operation. There is no
  window between "is the version still 2?" and "write the record" where a third party could sneak in.
  You cannot get this guarantee by reading the version yourself and then writing — that's two round
  trips with a gap, and the gap is where lost updates live.
- **The retry re-reads, not re-writes.** On conflict, the loop goes back to STEP 1 and fetches a fresh
  copy — it does *not* just resubmit the stale local record. That's what lets the losing agent pick up
  the winner's changes before applying its own.
- **The agents never know about each other.** Pricing doesn't check whether ETA is running. Each one
  just reads, modifies, and conditionally writes. DynamoDB arbitrates. This is the property the lesson
  keeps emphasizing, and it's what makes the pattern scale — adding a fourth agent requires no
  coordination code.

### Why the demo uses `put_item`, not `update_item` — and why that's deliberate

The file's own header comment shows the production form:

``` Python
table.update_item(
    Key={'trip_id': trip_id},
    UpdateExpression='SET driver = :d, version = version + :one',
    ConditionExpression='version = :expected',
    ExpressionAttributeValues={':d': driver_info, ':expected': N, ':one': 1},
)
```

`update_item` with `SET driver = :d` touches **one attribute**. Even without the version condition, it
couldn't clobber `fare` — it never writes `fare`. The demo instead uses `put_item` of the **whole
record**, which *would* clobber every other field if the read was stale. That is exactly the lost-update
scenario — and it means the `ConditionExpression` is the **only** thing standing between the demo and
data loss. The demo picks the more dangerous primitive on purpose, so the lock is visibly load-bearing
rather than belt-and-braces. For real code, prefer `update_item` on specific attributes *and* keep the
version condition; the two together are stronger than either alone.

### `to_dynamo` / `from_dynamo` — the `Decimal` tax

Two small helpers wrap every read and write:

``` Python
def to_dynamo(obj):     # Python → DynamoDB: floats become Decimal
    return json.loads(json.dumps(obj), parse_float=Decimal)

def from_dynamo(obj):   # DynamoDB → Python: Decimal becomes int or float
    return json.loads(json.dumps(obj, default=_dynamo_default))
```

boto3 refuses Python `float` for DynamoDB numbers and demands `Decimal`. Module 5 dodged this by
storing numbers as strings; this module does it properly, round-tripping through JSON to coerce types
at the boundary. The `_dynamo_default` helper turns a whole-number `Decimal` back into `int` and
anything else into `float`, so the application never sees `Decimal` objects.

## The agents

Three workers, the Module 1 shape, all Nova Lite at `temperature=0.0`. Each tool does the same three
things: read the trip with `get_trip`, compute its slice, write it with `update_trip`.

| Agent | Tool | Writes | Logic |
|---|---|---|---|
| DriverMatchAgent | `match_driver` | `driver` | preferred driver from memory if available, else highest-rated |
| PricingAgent | `calculate_fare` | `fare` | `base + per_mile × miles × surge`; miles = 8.5 if "Airport" appears in pickup or destination, else 5.2 |
| ETAAgent | `calculate_eta` | `eta`, **then `status`** | pickup 4 min (premium) or 7 (standard) + trip 25 (airport) or 15 |

> **Note:** **the ETA agent writes twice.** `calculate_eta` calls `update_trip` once for the `eta`
> fields and a *second* time for `{"status": "confirmed"}`. So a full trip is **four writes, not
> three**, and the sequential scenario ends at **version 4**, not version 3 as the narration says
> ("version went from 0 to 3, one increment per write" — three agents, but four increments). It also
> means Scenario 2 has four concurrent writes contending for one record, more conflict pressure than
> the lesson describes.

## Cross-session memory — `rider_memory` walked through

The DriverMatchAgent's tool is where the second kind of state lives. Here is the relevant part of
`match_driver`:

``` Python
rider_memory = {}        # module-level, defined once at import

@tool
def match_driver(trip_id: str) -> str:
    trip = get_trip(trip_id)
    ride_type = trip.get("ride_type", "standard")
    drivers = AVAILABLE_DRIVERS.get(ride_type, AVAILABLE_DRIVERS["standard"])
    rider_id = trip.get("rider_id")

    # ── READ from memory: does this rider have a saved preference? ──────────
    preferred = rider_memory.get(rider_id, {}).get("preferred_driver")

    if preferred and any(d["driver_id"] == preferred for d in drivers):
        # Preferred driver exists AND is in the available pool for this ride type
        best = next(d for d in drivers if d["driver_id"] == preferred)
        match_reason = "preferred driver (from memory)"
    else:
        # First-time rider, or preferred driver not available → fall back to rating
        best = max(drivers, key=lambda d: d["rating"])
        match_reason = "highest rated available"

    update_trip(trip_id, {"driver": {...best..., "match_reason": match_reason}})

    # ── WRITE to memory: remember this choice for next time ─────────────────
    if rider_id not in rider_memory:
        rider_memory[rider_id] = {}
    rider_memory[rider_id]["preferred_driver"] = best["driver_id"]
    rider_memory[rider_id]["last_driver_name"] = best["name"]
    ...
```

**Trace it across the three scenarios.** Alice (`RIDER-42`) rides in TRIP-001 and TRIP-003; Bob in
TRIP-002.

- **TRIP-001 (Alice, premium).** `rider_memory` is empty, so `preferred` is `None`. Falls to the
  `else` branch: highest-rated premium driver is Carlos Rivera (4.95). Writes
  `rider_memory["RIDER-42"] = {preferred_driver: "DRV-101", last_driver_name: "Carlos Rivera"}`.
  Match reason: *highest rated available*.
- **TRIP-002 (Bob, standard).** Bob has no memory entry. Highest-rated standard driver is Sarah Kim
  (4.85). Writes Bob's preference. Doesn't touch Alice's.
- **TRIP-003 (Alice, premium).** Now `rider_memory.get("RIDER-42")` returns Alice's entry and
  `preferred = "DRV-101"`. The `any(...)` check confirms DRV-101 is in the premium pool. Takes the
  `if` branch: Carlos Rivera again, match reason *preferred driver (from memory)*. **Same driver as
  TRIP-001** — the cross-session recall the demo is showing off.

The guard `any(d["driver_id"] == preferred for d in drivers)` matters: if Alice's preferred driver were
a *standard* driver and she booked a *premium* trip, the preference would be ignored and the rating
fallback would run. Memory informs the choice; it doesn't override availability.

### What this memory actually is — and what it isn't

**`rider_memory` is a Python dict.** The file says so directly: *"AgentCore Memory simulation
(in-memory; production uses bedrock-agentcore-control)."* The demo description's claim that the system
"uses AgentCore Memory to persist user preferences" is **not true of this code**; the lesson text's
"stand-in" framing is the accurate one. Alice's preference is remembered in Scenario 3 **only because
all three scenarios run in one process** and the dict is still alive. Stop the script and start it
again, and `rider_memory` is `{}` — Alice is a stranger. The demo demonstrates the *access pattern* of
cross-session memory (read preference → act → write preference back) without any of the persistence.

What it is standing in for: AgentCore Memory's `SESSION_SUMMARY` strategy, which would hold this same
preference in a managed store that survives process restarts, scales across instances, and is
retrieved by rider identity. The read/write shape of the tool would be identical; the only change is
that `rider_memory.get(...)` and `rider_memory[rider_id] = ...` become service calls.

> **Note:** this is the same move Module 3 and 4 made with hardcoded findings — demonstrate the
> *pattern* with the simplest thing that exhibits it, and defer the real integration. Reasonable for
> teaching, but the gap between "a dict" and "a managed memory service" is the entire difficulty of
> cross-session state, so don't mistake the demo for having shown it solved.

## The three scenarios

| Scenario | Trip | What it tests | Expected |
|---|---|---|---|
| 1 — Sequential | TRIP-001, Alice | agents run one at a time; version increments cleanly | version 0 → 4 (four writes), zero conflicts |
| 2 — Concurrent | TRIP-002, Bob | all three agents in a `ThreadPoolExecutor`; writes collide | **conflicts > 0**, all resolved by retry, final record correct |
| 3 — Memory | TRIP-003, Alice again | the DriverMatch tool recalls Alice's preference | same driver as TRIP-001, reason *"preferred driver (from memory)"* |

Scenario 2 is the one that proves the locking works. The narration reports "something like conflicts =
2," but the actual number is non-deterministic — it depends on exactly when each thread's read and
write land relative to the others. It can be 0 if the three agents happen not to overlap (unlikely but
possible), and higher with four writes in flight. **What's deterministic is the final state:** driver,
fare, eta, and status all present, version equal to the number of successful writes. The conflict
count is diagnostic; the correctness of the final record is the result.

## Four things to carry forward

1. **Two kinds of state, two stores.** Within-session transactional state and cross-session
   conversational context have different lifetimes and consistency needs. DynamoDB for one, AgentCore
   Memory for the other. Don't make one store do both.
2. **A lost update is silent.** Two full-record writes from stale reads, and the last writer erases the
   first with no error. The version-conditioned write is what turns that silent loss into a loud,
   retryable `ConditionalCheckFailedException`.
3. **On conflict, re-read — don't re-send.** The retry loop goes back to the READ step so the loser
   picks up the winner's changes before applying its own. Resubmitting the stale local copy would just
   be a slower lost update.
4. **Compensating actions are the failure strategy you plan for before the first write.** If a
   workflow mutates external state and a later step fails, something has to undo the earlier steps.
   This module names it and doesn't build it; the capstone's Refund agent will need it.

## Key terms

- **Optimistic locking** — concurrency control that assumes conflicts are rare: read a version, write
  conditionally on that version still being current, retry on mismatch. No locks held up front.
- **Lost update** — the failure mode optimistic locking prevents: two writers read the same state and
  the second write silently overwrites the first.
- **Conditional expression** — a DynamoDB rule (`ConditionExpression`) that must evaluate true for a
  write to succeed, checked atomically on the server. The mechanism that makes optimistic locking work.
- **`ConditionalCheckFailedException`** — the error DynamoDB raises when the condition is false. In
  this pattern it means *someone wrote first; re-read and retry*, not *something is broken*.
- **Exponential backoff** — increasing the wait between retries (0.1s, 0.2s, 0.4s here) so contending
  writers spread out instead of colliding again immediately.
- **Compensating action** — an operation that undoes a previously committed step when a later step
  fails, to avoid an inconsistent state (e.g., refund a charge for a service never provisioned).
- **Human-in-the-loop escalation** — pausing an automated workflow and handing a decision to a person,
  with full context attached, when the error is too complex or critical to automate.
