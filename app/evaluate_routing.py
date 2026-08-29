from ai_sdk.agents import CapabilityRouter
from ai_sdk.evaluation import (
    DEFAULT_ROUTE_EVAL_CASES,
    RouteEvaluationRunner,
)


def main() -> None:
    report = RouteEvaluationRunner(
        CapabilityRouter(),
        minimum_accuracy=1.0,
    ).evaluate(DEFAULT_ROUTE_EVAL_CASES)

    print("Super AI route evaluation")
    print(
        f"Accuracy: {report.accuracy:.1%} "
        f"({report.correct_cases}/{report.total_cases})"
    )
    print(
        "Estimated model requests: "
        f"{report.estimated_model_requests}"
    )
    print(
        "Full-route baseline requests: "
        f"{report.full_route_baseline_requests}"
    )
    print(
        "Estimated requests saved: "
        f"{report.estimated_request_savings}"
    )
    print(
        "Mean local routing latency: "
        f"{report.mean_routing_latency_ms:.3f} ms"
    )
    print(f"Passed: {report.passed}")

    if report.failed_case_ids:
        print(
            "Failed cases: "
            + ", ".join(report.failed_case_ids)
        )


if __name__ == "__main__":
    main()
