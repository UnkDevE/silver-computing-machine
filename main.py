import sys

from src import testing_and_output as to

if __name__ == "__main__":
    if len(sys.argv) > 3:
        # set download to false so we don't download more datasets
        to.model_test_batch("./datasets", int(sys.argv[1]),
                            [sys.argv[2], sys.argv[3]], download=True)
    else:
        print("""args - (1) resolution set resolution for image downscaling,
                 will  be in square format i.e. 225 means a 225x225 image
                 (2) - model name to use for testing and training
                 (3) - weights for that model to use
              """)
