from collections.abc import Callable, Sequence
from pathlib import Path

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.application.rag_response import Citation
from ai_sdk.config import (
    CHAT_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_TOKEN_BUDGET,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    VECTOR_STORE_FILE,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.context.window import SlidingContextWindow
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingClient,
)
from ai_sdk.ingestion import (
    DocumentIngestor,
    DirectorySynchronizer,
    TextDocumentLoader,
    create_default_ingestor,
)
from ai_sdk.llm.claude import ClaudeClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.json_store import JsonVectorStore
from ai_sdk.storage.json import (
    JsonConversationRepository,
)


def print_history(conversation: Conversation) -> None:
    if conversation.is_empty():
        print("\nHistory is empty.\n")
        return

    print()

    for message in conversation.history():
        print(
            f"{message.role.capitalize()}: "
            f"{message.content}"
        )
        print(f"ID: {message.id}")
        print()


def print_documents(
    manager: RAGConversationManager,
) -> None:
    documents = manager.document_catalog()

    if not documents:
        print("\nDocument index is empty.\n")
        return

    print("\nIndexed documents:")

    for document in documents:
        print(
            f"- {document.document_id} | "
            f"chunks={document.chunk_count} | "
            f"source={document.source}"
        )

    print()


def print_citations(
    citations: Sequence[Citation],
) -> None:
    if not citations:
        print()
        return

    print("Sources:")

    for citation in citations:
        print(
            f"[{citation.position}] {citation.source} | "
            f"document={citation.document_id} | "
            f"chunk={citation.chunk_id} | "
            f"score={citation.score:.3f}"
        )

    print()


def print_help() -> None:
    print(
        "Commands:\n"
        "  /index <path>       Index a file or sync a directory\n"
        "  /documents          Show the document catalog\n"
        "  /remove <document>  Remove a document from the index\n"
        "  /history            Show conversation history\n"
        "  /save               Save conversation history\n"
        "  /clear              Clear conversation history\n"
        "  /help               Show commands\n"
        "  /exit               Exit\n"
    )


def load_document(file_path: str) -> Document:
    return TextDocumentLoader().load(
        Path(file_path)
    )


def load_documents(
    path: str,
) -> list[Document]:
    return create_default_ingestor().ingest(path)


def build_manager() -> RAGConversationManager:
    repository = JsonConversationRepository(
        CHAT_FILE
    )
    conversation = repository.load()
    embedding_client = (
        SentenceTransformerEmbeddingClient(
            model_name=EMBEDDING_MODEL
        )
    )
    retriever = HybridRetriever(
        embedding_client=embedding_client,
        vector_store=JsonVectorStore(
            VECTOR_STORE_FILE
        ),
    )

    return RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(
            conversation,
            context_window=SlidingContextWindow(
                max_tokens=CONTEXT_TOKEN_BUDGET
            ),
        ),
        client=ClaudeClient(),
        repository=repository,
        chunker=TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ),
        retriever=retriever,
        retrieval_k=RETRIEVAL_K,
    )


def run_cli(
    manager: RAGConversationManager,
    input_fn: Callable[[str], str] = input,
    ingestor: DocumentIngestor | None = None,
) -> None:
    ingestor = ingestor or create_default_ingestor()
    print("Claude RAG Chat")
    print_help()

    while True:
        try:
            prompt = input_fn("You: ").strip()

            if not prompt:
                continue

            command, _, argument = prompt.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command == "/exit":
                break

            if command == "/help":
                print_help()
                continue

            if command == "/save":
                manager.repository.save(
                    manager.conversation
                )
                print("Conversation saved.\n")
                continue

            if command == "/clear":
                manager.conversation.clear()
                manager.repository.save(
                    manager.conversation
                )
                print("Conversation cleared.\n")
                continue

            if command == "/history":
                print_history(manager.conversation)
                continue

            if command == "/index":
                if not argument:
                    print("Usage: /index <path>\n")
                    continue

                index_path = Path(argument).expanduser()

                if index_path.is_dir():
                    result = DirectorySynchronizer(
                        ingestor,
                        manager,
                    ).sync(index_path)
                    print(
                        f"Synchronized {result.root}: "
                        f"indexed={len(result.indexed_documents)}, "
                        f"unchanged={len(result.unchanged_documents)}, "
                        f"removed={len(result.removed_documents)}, "
                        f"chunks={result.indexed_chunks}.\n"
                    )
                    continue

                document = ingestor.ingest(index_path)[0]
                chunks = manager.index_document(document)
                print(
                    f"Indexed {document.id}: "
                    f"{len(chunks)} chunks."
                )

                print()
                continue

            if command == "/documents":
                print_documents(manager)
                continue

            if command == "/remove":
                if not argument:
                    print(
                        "Usage: /remove <document_id>\n"
                    )
                    continue

                deleted_count = manager.delete_document(
                    argument
                )

                if deleted_count:
                    print(
                        f"Removed {argument}: "
                        f"{deleted_count} chunks.\n"
                    )
                else:
                    print(
                        f"Document not found: {argument}\n"
                    )

                continue

            if command.startswith("/"):
                print(
                    f"Unknown command: {command}. "
                    "Use /help.\n"
                )
                continue

            print(
                "\nClaude: ",
                end="",
                flush=True,
            )

            for chunk in manager.stream_message(prompt):
                print(
                    chunk,
                    end="",
                    flush=True,
                )

            print()
            print_citations(manager.last_citations)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        except Exception as error:
            print(f"\nError: {error}\n")


def main() -> None:
    run_cli(build_manager())


if __name__ == "__main__":
    main()
