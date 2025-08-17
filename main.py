import sys
from src import testing_and_output as to

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            to.test_main(sys.argv[1])
    except FileNotFoundError:
        print("""file not found,
                  please give list of datasets to test from""")
