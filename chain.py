# this is Augmentation and Generation for RAG
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from retriever import get_shared_retriever
from retriever import get_patient_retriever

retriever = get_shared_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a nutrition information assistant. Answer the question using "
     "ONLY the context provided below. If the context doesn't contain enough "
     "information to answer, say so instead of guessing. This is for "
     "informational purposes only and is not medical advice.\n\n"
     "Patient Info:\n{patient_info}\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])
def format_context(nodes):
  # takes in list of nodes and turnsa them into a single string
    return "\n\n".join(node.text for node in nodes)

history = []
MAX_TURNS = 5
def ask(question, patient_id):
    nodes = retriever.retrieve(question)
    context = format_context(nodes)

    patient_retriever = get_patient_retriever(patient_id)
    patient_nodes = patient_retriever.retrieve(question)
    patient_context = format_context(patient_nodes)

    messages = prompt.invoke({
        "context": context,
        "question": question,
        "patient_info": patient_context,
        "history": history
    })
    response = llm.invoke(messages)
    # after the response, add the question and answer to the history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))
    history[:] = history[-(MAX_TURNS * 2):]
    return response.content


if __name__ == "__main__":
    patient_id = "jane_demo"
    questions = [
        "Reccomended potassium intake for me?",
        "What about without my conditon?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(ask(q, patient_id))
        print(f"(history length: {len(history)})")