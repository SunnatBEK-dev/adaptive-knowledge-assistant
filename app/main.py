from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.config import (
    CHAT_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    VECTOR_STORE_FILE,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingClient,
)
from ai_sdk.llm.claude import ClaudeClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.json_store import JsonVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever
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
    document_ids = manager.list_documents()

    if not document_ids:
        print("\nDocument index is empty.\n")
        return

    print("\nIndexed documents:")

    for document_id in document_ids:
        print(f"- {document_id}")

    print()


def print_help() -> None:
    print(
        "Commands:\n"
        "  /index <path>       Index or re-index a UTF-8 text file\n"
        "  /documents          List indexed document IDs\n"
        "  /remove <document>  Remove a document from the index\n"
        "  /history            Show conversation history\n"
        "  /save               Save conversation history\n"
        "  /clear              Clear conversation history\n"
        "  /help               Show commands\n"
        "  /exit               Exit\n"
    )


def load_document(file_path: str) -> Document:
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Document file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Document path is not a file: {path}"
        )

    resolved_path = path.resolve()

    try:
        content = resolved_path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            "Document must be a UTF-8 text file."
        ) from error

    path_digest = sha256(
        str(resolved_path).encode("utf-8")
    ).hexdigest()[:12]

    return Document(
        id=f"doc_{path_digest}",
        content=content,
        metadata={"source": str(resolved_path)},
    )


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
    retriever = SemanticRetriever(
        embedding_client=embedding_client,
        vector_store=JsonVectorStore(
            VECTOR_STORE_FILE
        ),
    )

    return RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
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
) -> None:
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

                document = load_document(argument)
                chunks = manager.index_document(document)
                print(
                    f"Indexed {document.id}: "
                    f"{len(chunks)} chunks.\n"
                )
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

            print("\n")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        except Exception as error:
            print(f"\nError: {error}\n")


def main() -> None:
    run_cli(build_manager())


if __name__ == "__main__":
    main()
