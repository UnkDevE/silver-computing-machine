import sys

from src import testing_and_output as to

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # set download to false so we don't download more datasets
        to.model_test_batch("./datasets", int(sys.argv[1]), download=True)
    else:
        print("""args - resolution set resolution for image downscaling, will
               be in square format i.e. 225 means a 225x225 image""")
