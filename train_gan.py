from modules.gan_lseg_module import GAN_LSegModule
from utils import do_training, get_default_argument_parser
import os
os.environ["WANDB_INIT_TIMEOUT"] = "3000"
os.environ["WANDB__SERVICE_WAIT"] = "3000"

if __name__ == "__main__":
    parser = GAN_LSegModule.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()
    do_training(args, GAN_LSegModule)
