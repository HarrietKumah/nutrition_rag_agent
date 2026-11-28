PROJECT BRIEF: Condition-Aware Nutrition Advisor

WHAT THIS IS
A portfolio/demo project for job applications (targeting AI Product Manager,
founding engineer, and health tech roles). The goal is to demonstrate hands-on
experience with RAG (retrieval-augmented generation) and agentic orchestration
frameworks in a health tech context — not to build a production medical tool.

WHAT THE APP DOES
A chat-based tool supporting multiple patient profiles (no real authentication —
just a lightweight "create new profile" flow). Each profile has its own library
of medical info: EHR-style background (conditions, allergies, history) plus lab
results that accumulate over time. On profile setup (and whenever new lab
results are added), uploaded documents are parsed and indexed into that
patient's own private vector store — fully isolated from other patients' data
and from the shared guideline corpus (multi-tenant RAG).

When chatting, the agent answers nutrition questions by combining: (1) the
shared guideline knowledge base (indexed medical nutrition guideline documents,
starting with CKD), (2) the active patient's own indexed library (EHR + lab
history), and (3) conversation memory for follow-ups in the same thread.
Lab-result retrieval uses metadata filtering (doc_type, date) rather than pure
semantic similarity, since recency matters more than lexical/semantic closeness
for questions like "what's my most recent potassium level."

This is explicitly a PROTOTYPE, not a medical advice tool. It should include a
clear disclaimer in the UI, and should avoid presenting anything as a diagnosis
or prescriptive medical instruction — it surfaces and grounds relevant guideline
information, framed as informational only.

TECH STACK
- LlamaIndex — document loading, chunking, indexing, and retrieval
- ChromaDB — vector database (running locally, no cloud account needed).
  Two kinds of collections: one shared `health_guidelines` collection for
  medical guideline docs, and one private collection per patient profile for
  their EHR/lab library. Never mixed — this is the multi-tenant isolation
  boundary.
- LangChain — orchestration layer: takes user question + retrieved chunks +
  conversation history, generates grounded responses
- Frontend: simple chat UI (framework TBD — keep it minimal, this is a backend/
  architecture-focused demo, not a polished consumer app)

BUILD SEQUENCE (please follow this order — don't skip ahead to the UI)
1. Load the CKD guideline PDF (in docs/ folder), chunk it with LlamaIndex,
   index it into Chroma. Test: query "what is the sodium limit for CKD
   patients?" and confirm it retrieves the right chunk (source doc says
   ~2,300 mg/day). DONE.
2. Wrap retrieval in a LangChain chain: takes a question, retrieves relevant
   chunks, generates a grounded answer. Test via terminal script with a
   hardcoded question — no UI yet. DONE.
3. Add a document parsing step: take sample/hardcoded patient info (condition
   + lab values) and feed it into the chain alongside retrieval. DONE as a
   stepping stone (one hardcoded patient file injected directly into the
   prompt) — now being superseded by step 5's per-patient vector-store
   architecture to support multiple profiles and accumulating lab history.
4. Add conversation memory so follow-up questions work in a thread. DONE
   (sliding-window history, see chain.py).
5. Build the per-patient data architecture: file upload + parsing (reusing
   LlamaIndex's PDF/text readers, same as step 1) at profile-setup time,
   indexing into a private per-patient Chroma collection. Tag lab-result
   chunks with metadata (doc_type, date) so lab-specific queries can filter
   and sort by recency instead of relying on semantic similarity alone.
   Update chain.py to retrieve from both the shared guideline collection and
   the active patient's own collection, merging both into context.
6. Build the chat UI: multi-profile support (create-new-profile flow, profile
   switcher, EHR/lab upload), a generated summary on profile load, chat with
   follow-up questions, and a visible "informational only" disclaimer.



NOTE: The knowledge base will expand to other conditions (diabetes,
hypertension, etc.) later — the architecture should be reasonably easy to
extend with additional documents, but don't over-engineer for that now.
Prove it works with one document first.
