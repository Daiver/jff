import mlflow
import os
import time
from mlflow import log_metric, log_param, log_artifact


if __name__ == "__main__":
    mlflow.set_experiment("First")
    with mlflow.start_run():
        # Log a parameter (key-value pair)
        log_param("param1", 5)

        # Log a metric; metrics can be updated throughout the run
        for i in range(200):
            time.sleep(0.1)
            log_metric("foo1", 1 * i)
            log_metric("foo2", 2 * i)
            log_metric("foo3", 3 * i)
            log_metric("foo4", 3 * i)
            log_metric("foo5", 3 * i)
            log_metric("foo6", 3 * i)
            log_metric("foo7", 3 * i)
            log_metric("foo8", 3 * i)
            log_metric("foo9", 3 * i)
            log_metric("foo10", 3 * i)
            log_metric("foo11", 3 * i)
            log_metric("foo12", 3 * i)
            log_metric("foo13", 3 * i)
            log_metric("foo14", 3 * i)
            log_metric("foo15", 3 * i)
            log_metric("foo16", 3 * i)
            log_metric("foo17", 3 * i)
            log_metric("foo18", 3 * i)
            log_metric("foo19", 3 * i)
            log_metric("foo20", 3 * i)

            # Log an artifact (output file)
            with open("output.txt", "w") as f:
                f.write("Hello world!")
            with open("output.txt", "w") as f:
                f.write("Hello world!{}".format(i))
            log_artifact("output.txt")
