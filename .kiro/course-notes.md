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