import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args
import wandb


def main():
    # wandb.init(project="Expert_Demos_Cogvideo2", name="cogvideox_i2v_full_MuJoCo", id="8ma2h0e7", resume="must", reinit=True)
    # wandb.define_metric("global_step")
    # wandb.define_metric("*", step_metric="global_step")
    args = Args.parse_args()
    wandb.init(project="Expert_Demos_Cogvideo3", name=args.wandb_name)
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()
    wandb.finish()


if __name__ == "__main__":
    main()
