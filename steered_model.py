from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from steering_hooks import init_model_hook_steering, load_steering_vec


@register_model("steered_hf")
class SteeredHFLM(HFLM):
    def __init__(self, *args, steering_vec_file=None, control_type="none",
                 control_num=0, control_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_dict = None
        if steering_vec_file and control_type != "none":
            self.hook_dict = init_model_hook_steering(self.model)
            self.hook_dict[int(control_num)] = load_steering_vec(
                steering_vec_file, int(control_num), float(control_scale)
            )


if __name__ == "__main__":
    cli_evaluate()
