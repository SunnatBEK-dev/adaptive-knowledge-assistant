from ai_sdk.config import PROJECT_ROOT  # noqa: F401
from ai_sdk.readiness import inspect_ai_readiness


def main() -> None:
    report = inspect_ai_readiness()
    print("AI configuration readiness (no network requests):")
    for provider in report.providers:
        status = "READY" if provider.ready else "NOT READY"
        detail = ""
        if provider.missing_variables:
            detail = (
                " | missing: "
                + ", ".join(provider.missing_variables)
            )
        print(f"- {provider.display_name}: {status}{detail}")

    ready_count = len(report.direct_chat_ready_providers)
    print(
        "Direct Chat ready providers: "
        f"{ready_count}/{len(report.providers)}"
    )
    super_status = "READY" if report.super_ai_ready else "NOT READY"
    print(f"Super AI: {super_status}")


if __name__ == "__main__":
    main()
