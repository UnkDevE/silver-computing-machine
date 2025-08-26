import sys

from src import testing_and_output as to

if __name__ == "__main__":
    if len(sys.argv) > 5:
        # set download to false so we don't download more datasets
        print(sys.argv[2:])
        to.model_test_batch("./datasets", int(sys.argv[1]), int(sys.argv[2]),
                            sys.argv[3:], download=True)
    else:
        print("""args - (1) resolution set resolution for image downscaling,
                 will  be in square format i.e. 225 means a 225x225 image
                 (2) - training rounds model is to preform
                 (3) - model name to use for testing and training
                 (4) - weights for that model to use
                 (5) - loss function for model to use
                 (6) - optimizer for training to use
              """)
