# this is Augmentation and Generation for RAG
from langchain_openai import ChatOpenAI
from retriever import get_retriever

retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a nutrition information assistant. Answer the question using "
     "ONLY the context provided below. If the context doesn't contain enough "
     "information to answer, say so instead of guessing. This is for "
     "informational purposes only and is not medical advice.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])
def format_context(nodes):
  # takes in list of nodes and turnsa them into a single string
    return "\n\n".join(node.text for node in nodes)


def ask(question):
    nodes = retriever.retrieve(question)
    context = format_context(nodes)
    messages = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    answer = ask("What are the guidelines for high fever?")
    print(answer)