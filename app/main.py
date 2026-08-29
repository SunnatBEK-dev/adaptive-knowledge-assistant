from collections.abc import Callable, Sequence
from pathlib import Path

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.application.rag_response import Citation
from ai_sdk.application.modes import ApplicationMode
from ai_sdk.agents import (
    CapabilityRouter,
    DependencyHandoffCoordinator,
    HandoffOutputFormat,
    HandoffStage,
    MultiAgentCoordinator,
    SuperAIRoute,
    create_provider_worker,
)
from ai_sdk.config import (
    CHAT_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_SUMMARY_TOKEN_BUDGET,
    CONTEXT_TOKEN_BUDGET,
    DIRECT_CHAT_DIR,
    EMBEDDING_MODEL,
    MEMORY_FILE,
    MEMORY_RETRIEVAL_K,
    RETRIEVAL_K,
    SUPER_AI_CHAT_FILE,
    VECTOR_STORE_FILE,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.context.summary import (
    ExtractiveConversationSummarizer,
)
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
from ai_sdk.llm.factory import (
    create_llm_client,
    normalize_llm_provider,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.super_ai import (
    RoutedSuperAIClient,
    SuperAIClient,
)
from ai_sdk.memory.json_store import JsonMemoryStore
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


def print_memories(
    manager: RAGConversationManager,
) -> None:
    memories = manager.list_memories()

    if not memories:
        print("\nLong-term memory is empty.\n")
        return

    print("\nLong-term memories:")

    for memory in memories:
        print(f"- {memory.id} | {memory.content}")

    print()


def print_help() -> None:
    print(
        "Commands:\n"
        "  /index <path>       Index a file or sync a directory\n"
        "  /documents          Show the document catalog\n"
        "  /remove <document>  Remove a document from the index\n"
        "  /remember <fact>    Store a long-term memory\n"
        "  /memories           Show long-term memories\n"
        "  /forget <memory>    Remove a long-term memory\n"
        "  /history            Show conversation history\n"
        "  /stats              Show Super AI runtime statistics\n"
        "  /save               Save conversation history\n"
        "  /clear              Clear conversation history\n"
        "  /help               Show commands\n"
        "  /exit               Exit\n"
    )


def print_super_ai_stats(
    manager: RAGConversationManager,
) -> None:
    client = getattr(manager, "client", None)
    if not isinstance(client, RoutedSuperAIClient):
        print("\nRuntime statistics are available in Super AI mode.\n")
        return

    report = client.stats.report()
    if report.total_runs == 0:
        print("\nNo Super AI runs have been recorded yet.\n")
        return

    routes = ", ".join(
        f"{route}={count}"
        for route, count in sorted(report.route_counts.items())
    )
    stages = ", ".join(
        f"{stage}={count}"
        for stage, count in sorted(
            report.stage_execution_counts.items()
        )
    )
    print("\nSuper AI runtime statistics:")
    print(
        f"- Runs: {report.total_runs} "
        f"(successful={report.successful_runs}, "
        f"failed={report.failed_runs})"
    )
    print(f"- Routes: {routes or 'none'}")
    print(
        f"- Stages: executed={report.executed_stage_count}, "
        f"failed={report.failed_stage_count}, "
        f"blocked={report.blocked_stage_count}"
    )
    print(f"- Stage executions: {stages or 'none'}")
    print(
        f"- Mean duration: {report.mean_duration_ms:.1f} ms "
        f"(max={report.max_duration_ms:.1f} ms)\n"
    )


def load_document(file_path: str) -> Document:
    return TextDocumentLoader().load(
        Path(file_path)
    )


def load_documents(
    path: str,
) -> list[Document]:
    return create_default_ingestor().ingest(path)


def build_manager(
    *,
    provider: str | None = None,
    conversation_file: Path = CHAT_FILE,
    client: BaseLLMClient | None = None,
) -> RAGConversationManager:
    if provider is not None and client is not None:
        raise ValueError(
            "Configure either a provider or an explicit client."
        )
    if client is not None and not isinstance(
        client,
        BaseLLMClient,
    ):
        raise TypeError(
            "Explicit client must be a BaseLLMClient."
        )
    repository = JsonConversationRepository(
        conversation_file
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
            summary_memory=(
                ExtractiveConversationSummarizer(
                    max_tokens=(
                        CONTEXT_SUMMARY_TOKEN_BUDGET
                    )
                )
            ),
        ),
        client=(
            create_llm_client(provider)
            if client is None
            else client
        ),
        repository=repository,
        chunker=TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ),
        retriever=retriever,
        retrieval_k=RETRIEVAL_K,
        memory_store=JsonMemoryStore(MEMORY_FILE),
        memory_retrieval_k=MEMORY_RETRIEVAL_K,
    )


def build_direct_chat_manager(
    provider: str,
    *,
    conversation_file: Path | None = None,
) -> RAGConversationManager:
    normalized = normalize_llm_provider(provider)
    chat_file = conversation_file or (
        DIRECT_CHAT_DIR / f"{normalized}.json"
    )
    return build_manager(
        provider=normalized,
        conversation_file=chat_file,
    )


def build_super_ai_manager(
    *,
    conversation_file: Path = SUPER_AI_CHAT_FILE,
) -> RAGConversationManager:
    context_worker = create_provider_worker(
        "context",
        "Extract relevant facts, constraints, and missing evidence",
        "gemini",
    )
    reasoning_worker = create_provider_worker(
        "reasoner",
        "Perform careful analysis using the available evidence",
        "anthropic",
    )
    synthesis_worker = create_provider_worker(
        "synthesizer",
        "Produce one clear final answer for the user",
        "openai",
    )
    coordinator = MultiAgentCoordinator([
        context_worker,
        reasoning_worker,
        synthesis_worker,
    ])

    def context_stage() -> HandoffStage:
        return HandoffStage(
            "context",
            "context",
            "Extract facts, constraints, and uncertainties. "
            "Do not invent missing evidence.",
            output_format=HandoffOutputFormat.STRUCTURED,
        )

    def reasoning_stage(
        *dependencies: str,
    ) -> HandoffStage:
        return HandoffStage(
            "reasoning",
            "reasoner",
            "Analyze the request and available evidence. "
            "Identify contradictions and a sound solution. "
            "Carry forward every useful verified fact.",
            output_format=HandoffOutputFormat.STRUCTURED,
            depends_on=dependencies,
        )

    def final_stage(*dependencies: str) -> HandoffStage:
        return HandoffStage(
            "final",
            "synthesizer",
            "Create the final answer from verified useful points. "
            "Do not mention internal stages unless needed.",
            depends_on=dependencies,
        )

    def workflow(
        stages: list[HandoffStage],
    ) -> SuperAIClient:
        return SuperAIClient(DependencyHandoffCoordinator(
            coordinator,
            stages,
        ))

    routed_client = RoutedSuperAIClient(
        CapabilityRouter(),
        {
            SuperAIRoute.FAST: workflow([
                final_stage(),
            ]),
            SuperAIRoute.CONTEXT: workflow([
                context_stage(),
                final_stage("context"),
            ]),
            SuperAIRoute.REASONING: workflow([
                reasoning_stage(),
                final_stage("reasoning"),
            ]),
            SuperAIRoute.FULL: workflow([
                context_stage(),
                reasoning_stage("context"),
                final_stage("context", "reasoning"),
            ]),
        },
    )
    return build_manager(
        conversation_file=conversation_file,
        client=routed_client,
    )


def select_application_mode(
    input_fn: Callable[[str], str] = input,
) -> ApplicationMode:
    choices = {
        "1": ApplicationMode.DIRECT_CHAT,
        "direct": ApplicationMode.DIRECT_CHAT,
        "direct_chat": ApplicationMode.DIRECT_CHAT,
        "2": ApplicationMode.SUPER_AI,
        "super": ApplicationMode.SUPER_AI,
        "super_ai": ApplicationMode.SUPER_AI,
    }
    print("Choose a section:")
    print("  1. Direct Chat")
    print("  2. Super AI")
    while True:
        selected = input_fn("Mode: ").strip().casefold()
        mode = choices.get(selected)
        if mode is not None:
            return mode
        print("Invalid mode. Choose 1 or 2.")


def select_direct_provider(
    input_fn: Callable[[str], str] = input,
) -> str:
    choices = {
        "1": "anthropic",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "2": "openai",
        "gpt": "openai",
        "openai": "openai",
        "3": "gemini",
        "gemini": "gemini",
    }
    print("Choose an AI provider:")
    print("  1. Claude (Anthropic)")
    print("  2. GPT (OpenAI)")
    print("  3. Gemini (Google)")
    while True:
        selected = input_fn("Provider: ").strip().casefold()
        provider = choices.get(selected)
        if provider is not None:
            return provider
        print("Invalid provider. Choose 1, 2, or 3.")


def run_cli(
    manager: RAGConversationManager,
    input_fn: Callable[[str], str] = input,
    ingestor: DocumentIngestor | None = None,
    *,
    title: str = "AI RAG Chat",
    stream_responses: bool = True,
) -> None:
    ingestor = ingestor or create_default_ingestor()
    print(title)
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

            if command == "/stats":
                print_super_ai_stats(manager)
                continue

            if command == "/remember":
                if not argument:
                    print("Usage: /remember <fact>\n")
                    continue

                memory = manager.remember(argument)
                print(f"Remembered {memory.id}.\n")
                continue

            if command == "/memories":
                print_memories(manager)
                continue

            if command == "/forget":
                if not argument:
                    print("Usage: /forget <memory_id>\n")
                    continue

                if manager.forget(argument):
                    print(f"Forgot {argument}.\n")
                else:
                    print(f"Memory not found: {argument}\n")

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

            if stream_responses:
                print(
                    "\nAssistant: ",
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
            else:
                response = manager.send_message(prompt)
                print(f"\nAssistant: {response}")
            print_citations(manager.last_citations)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        except Exception as error:
            print(f"\nError: {error}\n")


def main() -> None:
    mode = select_application_mode()
    if mode is ApplicationMode.DIRECT_CHAT:
        provider = select_direct_provider()
        run_cli(
            build_direct_chat_manager(provider),
            title=f"Direct Chat ({provider})",
        )
        return

    run_cli(
        build_super_ai_manager(),
        title="Super AI",
        stream_responses=False,
    )


if __name__ == "__main__":
    main()
