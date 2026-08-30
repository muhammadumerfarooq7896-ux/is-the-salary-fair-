# 💰 Is The Salary Fair?

An autonomous multi-agent system that goes out, finds real job postings,
figures out what each role should actually pay, and taps you on the
shoulder the moment it finds one that looks underpaid.

No manual searching. No spreadsheets. Just a team of AI agents doing the
legwork and reasoning through it themselves.

<p align="center">
  <img src="asset/output 1.png" width="48%">
  <img src="asset/output 2.png" width="48%">
</p>
---

## What's actually happening here

At its core, this project answers one question — *"is this job paying what
it should?"* — by combining three completely different prediction methods,
wrapping them in an autonomous decision-making loop, and giving the whole
thing a live dashboard to watch it work.

### 🎯 A fine-tuned language model, trained from scratch

Using **QLoRA** (a memory-efficient fine-tuning technique), a Llama-3.2
model was trained on thousands of real job postings and their salaries —
not prompted, not zero-shot, actually *trained* to understand the
relationship between a role description and fair pay. It's deployed as a
live, on-demand serverless endpoint, ready to answer in seconds.

**`finetuning using QLORA/llama32_qlora.ipynb`** — the full training run.
**`dataset prep/`** — the pipeline that builds and prepares the training
data before any fine-tuning happens.

### 🔍 An agentic RAG pipeline

Rather than guessing in a vacuum, the Frontier Agent retrieves the 5 most
similar real job postings from a vector database of historical listings,
and reasons over them: *"these comparable roles paid around $X, so this one
should too."* Retrieval-Augmented Generation, grounded in real data instead
of pure guesswork.

**`agents/frontier_agent.py`** — the RAG logic.
**`salary_vectorstore/`** — the Chroma vector database it searches.

### 🧠 A custom neural network — zero language models involved

A completely independent estimator: a from-scratch PyTorch residual neural
network, trained on hashed text features, with no LLM anywhere in the loop.
A third, mathematically distinct opinion — a check against blind spots the
two language-model-based estimators might share.

**`agents/neural_network_agent.py`** + **`deep_neural_network.pth`** — the
trained weights and inference logic.

### ⚖️ An ensemble that blends all three

One estimator alone is never fully trustworthy. The Ensemble Agent runs a
posting through the fine-tuned specialist, the RAG-based frontier model,
and the neural network — then blends their three independent answers into
one final, more robust estimate.

**`agents/ensemble_agent.py`**

### 🕵️ Multi-agent orchestration — genuinely autonomous, not scripted

This is the heart of the system. Instead of hardcoding "step 1, step 2,
step 3," an LLM is handed a set of real tools — *scan the internet*,
*estimate a fair salary*, *notify the user* — and left to decide, on its
own, what to do and in what order. It reasons through a live tool-calling
loop, executing real actions and adjusting its next move based on what it
finds, until it decides the job is done.

**`agents/scanner_agent.py`** — finds and filters real postings.
**`agents/preprocessor.py`** — cleans messy raw text before any estimator
sees it.
**`agents/messaging_agent.py`** — writes and sends the alert when something
worth flagging is found.
**`job_agent_framework.py`** — orchestrates the whole run, persists memory
of what's already been surfaced, and powers the live dashboard.

### 📊 A live dashboard to watch it all happen

**`app.py`** — a Gradio interface where you can watch every agent announce
what it's doing in real time, browse everything flagged so far, and see a
3D map of how job postings cluster in embedding space.

---

## Project structure

```
is the salary fair/
├── agents/
│   ├── agent.py                 # shared base class for logging
│   ├── specialist_agent.py      # fine-tuned model, deployed on Modal
│   ├── frontier_agent.py        # agentic RAG estimator
│   ├── neural_network_agent.py  # custom neural network estimator
│   ├── ensemble_agent.py        # blends all three estimates
│   ├── preprocessor.py          # cleans raw descriptions
│   ├── scanner_agent.py         # finds real postings
│   ├── messaging_agent.py       # sends alerts
│   ├── postings.py              # shared data schemas
│   ├── job_items.py             # training-data schema
│   └── evaluator.py             # offline accuracy evaluation
├── dataset prep/
│   ├── build_dataset.ipynb      # raw dataset construction
│   ├── job_items.py
│   └── salary_prep.ipynb        # tokenization + prompt building
├── finetuning using QLORA/
│   └── llama32_qlora.ipynb      # the actual fine-tuning run
├── salary_vectorstore/          # Chroma vector database
├── app.py                       # live dashboard
├── job_agent_framework.py       # top-level orchestrator
├── log_utils.py                 # log formatting for the dashboard
├── pricer_service.py            # Modal deployment for the fine-tuned model
├── deep_neural_network.pth      # trained neural network weights
├── memory.json                  # persisted history of flagged opportunities
└── .env                         # API keys and secrets
```

---

## How a single run works, end to end

1. **Scan** — real job postings are pulled from the internet and the 5
   clearest, salary-stated listings are selected.
2. **Estimate** — each posting is priced by the fine-tuned specialist, the
   RAG frontier agent, and the neural network, then blended into one
   ensemble estimate.
3. **Compare** — that estimate is checked against what the posting actually
   advertises.
4. **Decide** — the autonomous planning agent picks the most compelling
   underpaid posting and decides whether it's worth flagging.
5. **Notify** — a push notification goes out, written by an LLM to be
   short and easy to read.
6. **Remember** — what's been shown is saved, so nothing repeats.

All of it driven by one agent reasoning through the sequence itself, live.

---

## Getting started

1. Add your API keys and secrets to `.env`.
2. Deploy the fine-tuned model: `modal deploy pricer_service.py`
3. Install dependencies (Gradio, ChromaDB, sentence-transformers, PyTorch,
   Modal, litellm, and friends).
4. Run the dashboard: `python app.py`

Then just watch it work.
