import sys
from random import randint
import torch
import multiprocessing
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True, warn_only=True)

GENERATOR_SEED = randint(0, sys.maxsize)
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    from src import testing_and_output as to
    if len(sys.argv) > 7:
        # for reproduciblity purposes
        if sys.argv[8] != 0:
            GENERATOR_SEED = int(sys.argv[8])

        # needs to be _global_ here otherwise generation of seed will start at 0
        # multiple times
        torch.manual_seed(GENERATOR_SEED)
        torch.cuda.manual_seed(GENERATOR_SEED)
        print("MANUAL SANITY CHECK RANDOM SEED IS {}".format(str(GENERATOR_SEED)))
        print("REPRODUCEABLE RANDOM SEED IS: {}".format(str(torch.initial_seed())))

        to.model_test_batch("./datasets", int(sys.argv[1]), int(sys.argv[2]),
                            sys.argv[3:7], download=bool(int(sys.argv[7])),
                            seed=sys.argv[8])
    else:
        print("""args - (1) resolution set resolution for image downscaling,
                    will  be in square format i.e. 225 means a 225x225 image
                    (2) - training rounds model is to preform
                    (3) - model name to use for testing and training
                    (4) - weights for that model to use
                    (5) - loss function for model to use
                    (6) - optimizer for training to use
                    (7) - download models
                    (8) - seed for reproducibility purposes
                        set to zero for new seed
                """)
