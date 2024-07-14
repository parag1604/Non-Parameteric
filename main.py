from utils import ArgumentStorage, load_data, plot_decision_boundary
from model import KNNModel


def main(
        args: ArgumentStorage,
) -> None:
    X_train, X_test, y_train, y_test = load_data(test_ratio=args.test_ratio)

    model = KNNModel(k=args.k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.3f}")

    plot_decision_boundary(model, X_train, y_train, partition="train")
    plot_decision_boundary(model, X_test, y_test, partition="test")


if __name__ == '__main__':
    import sys
    args = ArgumentStorage({
        'k': int(sys.argv[1]) if len(sys.argv) > 1 else 2,
        'test_ratio': 0.2,
    })
    main(args)
