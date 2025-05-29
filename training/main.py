"""Main entry point for the ML pipeline."""

from prefect import flow

from hpo import run_optimization
from preprocess import run_data_prep
from register import run_register_model
from train import run_train


@flow
def main_flow(
    input_file: str = "./data/healthcare-dataset-stroke-data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    n_trials: int = 10,
    top_n: int = 5,
):
    """Main entry point for the ML pipeline."""
    run_data_prep(
        input_file=input_file,
        output_dir="./models",
        test_size=test_size,
        random_state=random_state,
    )
    run_train(
        data_path="./models",
    )
    run_optimization(
        data_path="./models",
        num_trials=n_trials,
    )
    run_register_model(
        top_n=top_n,
    )


if __name__ == "__main__":
    main_flow.serve(
        name="main-flow",
        parameters={
            "input_file": "./data/healthcare-dataset-stroke-data.csv",
            "test_size": 0.2,
            "random_state": 42,
            "n_trials": 10,
            "top_n": 5,
        },
    )
