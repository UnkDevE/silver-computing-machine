from src import testing_and_output as to

if __name__ == "__main__":
    # set download to false so we don't download more datasets
    to.model_test_batch("./datasets", download=False)
