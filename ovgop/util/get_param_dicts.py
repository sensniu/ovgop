"""Optimizer parameter groups for OVGOP training."""


def matches_any(name, keywords):
    return any(keyword in name for keyword in keywords)


def get_param_dict(args, model):
    if args.param_dict_type != "tuning":
        raise ValueError("This release only supports param_dict_type='tuning'")

    object_detection_modules = {
        "bert",
        "transformer",
        "feat_map",
        "input_proj",
        "backbone",
    }
    gaze_modules = {
        "temperature",
        "gatector",
        "clip_model",
        "scene_adapter",
        "face_adapter",
        "text_encoder",
    }

    named_parameters = list(model.named_parameters())
    known_modules = object_detection_modules | gaze_modules
    unknown_modules = sorted(
        {
            name.split(".")[0]
            for name, _ in named_parameters
            if name.split(".")[0] not in known_modules
        }
    )
    if unknown_modules:
        raise ValueError(f"Unassigned model modules: {unknown_modules}")

    def selected(module_names, *, exclude_backbone=False, exclude_projection=False):
        parameters = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad or name.split(".")[0] not in module_names:
                continue
            if exclude_backbone and matches_any(name, args.lr_backbone_names):
                continue
            if exclude_projection and matches_any(name, args.lr_linear_proj_names):
                continue
            if "text_encoder" in name:
                continue
            parameters.append(parameter)
        return parameters

    return [
        {
            "params": selected(
                object_detection_modules,
                exclude_backbone=True,
                exclude_projection=True,
            ),
            "lr": args.lr,
        },
        {
            "params": selected(
                gaze_modules,
                exclude_backbone=True,
                exclude_projection=True,
            ),
            "lr": args.lr,
        },
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and matches_any(name, args.lr_backbone_names)
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and matches_any(name, args.lr_linear_proj_names)
            ],
            "lr": args.lr_linear_proj_mult,
        },
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and "text_encoder" in name
            ],
            "lr": args.lr_text_encoder,
        },
    ]
