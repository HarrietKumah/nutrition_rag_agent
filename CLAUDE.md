PROJECT BRIEF: Condition-Aware Nutrition Advisor

WHAT THIS IS
A portfolio/demo project for job applications (targeting AI Product Manager,
founding engineer, and health tech roles). The goal is to demonstrate hands-on
experience with RAG (retrieval-augmented generation) and agentic orchestration
frameworks in a health tech context — not to build a production medical tool.

WHAT THE APP DOES
A chat-based tool where a user uploads or pastes info about a medical condition
(starting with chronic kidney disease) and recent lab values, and asks nutrition
questions. The app retrieves relevant guidance from indexed medical nutrition
guideline documents and generates answers grounded in that retrieved content —
not just from the LLM's general knowledge. Follow-up questions continue in the
same chat thread with conversation memory.

This is explicitly a PROTOTYPE, not a medical advice tool. It should include a
clear disclaimer in the UI, and should avoid presenting anything as a diagnosis
or prescriptive medical instruction — it surfaces and grounds relevant guideline
information, framed as informational only.

TECH STACK
- LlamaIndex — document loading, chunking, indexing, and retrieval
- ChromaDB — vector database (running locally, no cloud account needed)
- LangChain — orchestration layer: takes user question + retrieved chunks +
  conversation history, generates grounded responses
- Frontend: simple chat UI (framework TBD — keep it minimal, this is a backend/
  architecture-focused demo, not a polished consumer app)

BUILD SEQUENCE (please follow this order — don't skip ahead to the UI)
1. Load the CKD guideline PDF (in docs/ folder), chunk it with LlamaIndex,
   index it into Chroma. Test: query "what is the sodium limit for CKD
   patients?" and confirm it retrieves the right chunk (source doc says
   ~2,300 mg/day).
2. Wrap retrieval in a LangChain chain: takes a question, retrieves relevant
   chunks, generates a grounded answer. Test via terminal script with a
   hardcoded question — no UI yet.
3. Add a document parsing step: take sample/hardcoded patient info (condition
   + lab values) and feed it into the chain alongside retrieval.
4. Add conversation memory so follow-up questions work in a thread.
5. Only once 1–4 work end-to-end: build a simple chat UI — upload/enter
   patient info, see a generated summary, ask follow-up questions in chat.



NOTE: The knowledge base will expand to other conditions (diabetes,
hypertension, etc.) later — the architecture should be reasonably easy to
extend with additional documents, but don't over-engineer for that now.
Prove it works with one document first.
