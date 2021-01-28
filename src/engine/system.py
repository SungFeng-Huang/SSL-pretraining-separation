import torch
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


class GeneralSystem(System):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)

        # Load from checkpoint if provided.
        if self.config["main_args"].get("load_path") is not None:
            all_states = torch.load(self.config["main_args"]["load_path"], map_location="cpu")
            assert "state_dict" in all_states

            # If the checkpoint is not the serialized "best_model.pth", its keys 
            # would start with "model.", which should be removed to avoid none 
            # of the parameters are loaded.
            for key in list(all_states["state_dict"].keys()):
                if key.startswith("model"):
                    print(f"key {key} changed to {key.split('.', 1)[1]}")
                    all_states["state_dict"][key.split('.', 1)[1]] = all_states["state_dict"][key]
                    del all_states["state_dict"][key]

            # For debugging, set strict=True to check whether only the following
            # parameters have different sizes (since n_src=1 for pre-training
            # and n_src=2 for fine-tuning):
            # for ConvTasNet: "masker.mask_net.1.*"
            # for DPRNNTasNet/DPTNet: "masker.first_out.1.*"
            if self.config["main_args"]["model"] == "ConvTasNet":
                print(f"key masker.mask_net.1.* removed")
                del all_states["state_dict"]["masker.mask_net.1.weight"]
                del all_states["state_dict"]["masker.mask_net.1.bias"]
            elif self.config["main_args"]["model"] in ["DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet"]:
                print(f"key masker.first_out.1.* removed")
                del all_states["state_dict"]["masker.first_out.1.weight"]
                del all_states["state_dict"]["masker.first_out.1.bias"]
            self.model.load_state_dict(all_states["state_dict"], strict=False)
