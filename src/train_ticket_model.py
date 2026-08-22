import json

from src.torch_ticket_model import (
    evaluate_logistic_regression_baseline,
    train_multitask_model,
)


def main() -> None:
    torch_metrics = train_multitask_model()
    logistic_metrics = evaluate_logistic_regression_baseline()

    print("\nPyTorch multi-task classifier")
    print(json.dumps(torch_metrics, indent=2))
    print("\nLogistic Regression baseline")
    print(json.dumps(logistic_metrics, indent=2))


if __name__ == "__main__":
    main()
