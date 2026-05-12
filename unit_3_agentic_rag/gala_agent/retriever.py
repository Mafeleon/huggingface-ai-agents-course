from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.invoke(query)

        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])

        return "No matching guest information found."


def load_guest_dataset():
    guest_dataset = load_dataset("agents-course/unit3-invites", split="train")

    docs = [
        Document(
            page_content=(
                f"Name: {guest['name']}\n"
                f"Relation: {guest['relation']}\n"
                f"Description: {guest['description']}\n"
                f"Email: {guest['email']}"
            ),
            metadata={"name": guest["name"]},
        )
        for guest in guest_dataset
    ]

    return GuestInfoRetrieverTool(docs)