"""Gradient-Informed Selection Tuning (GIST)."""

import torch


def _layer_prefixes():
    prefixes = []
    for index in range(6):
        prefixes.append(f"transformer.encoder.layers.{index}")
    for index in range(6):
        prefixes.append(f"transformer.encoder.text_layers.{index}")
    for index in range(6):
        prefixes.append(f"transformer.encoder.fusion_layers.{index}")
    for index in range(6):
        prefixes.append(f"transformer.decoder.layers.{index}")
    return tuple(prefixes)


GIST_LAYER_PREFIXES = _layer_prefixes()


def selection_count(candidate_count, selection_ratio):
    if not 0 < selection_ratio <= 1:
        raise ValueError(f"GIST selection ratio must be in (0, 1], got {selection_ratio}")
    if candidate_count == 0:
        raise ValueError("Cannot select GIST parameters from an empty layer")
    return max(1, int(candidate_count * selection_ratio + 0.5))


def apply_selection(model, selection):
    mask = {
        parameter_name: trainable
        for layer_selection in selection.values()
        for parameter_name, trainable in layer_selection.items()
    }
    model_parameters = dict(model.named_parameters())
    missing = sorted(set(mask).difference(model_parameters))
    if missing:
        raise KeyError(f"GIST selection contains unknown parameters: {missing[:5]}")
    for parameter_name, trainable in mask.items():
        model_parameters[parameter_name].requires_grad = bool(trainable)


def select_sensitive_parameters(model, optimizer, selection_ratio):
    """Rank parameter tensors by AdamW's bias-corrected second moment."""
    scores = {layer_name: {} for layer_name in GIST_LAYER_PREFIXES}
    parameter_groups = {
        parameter: group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    for parameter_name, parameter in model.named_parameters():
        if parameter not in optimizer.state or parameter not in parameter_groups:
            continue
        for layer_name in GIST_LAYER_PREFIXES:
            if layer_name not in parameter_name:
                continue
            state = optimizer.state[parameter]
            if "exp_avg_sq" not in state or "step" not in state:
                break
            step = state["step"]
            if torch.is_tensor(step):
                step = step.item()
            beta2 = parameter_groups[parameter]["betas"][1]
            bias_correction = 1 - beta2 ** step
            if bias_correction > 0:
                score = (state["exp_avg_sq"] / bias_correction).mean().item()
                scores[layer_name][parameter_name] = score
            break

    sorted_scores = {
        layer_name: sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
        for layer_name, layer_scores in scores.items()
    }
    selection = {}
    for layer_name, layer_scores in sorted_scores.items():
        top_k = selection_count(len(layer_scores), selection_ratio)
        selection[layer_name] = {
            parameter_name: index < top_k
            for index, (parameter_name, _) in enumerate(layer_scores)
        }
    return selection, sorted_scores


def validate_selection(selection, selection_ratio):
    if set(selection) != set(GIST_LAYER_PREFIXES):
        missing = sorted(set(GIST_LAYER_PREFIXES).difference(selection))
        unexpected = sorted(set(selection).difference(GIST_LAYER_PREFIXES))
        raise ValueError(
            f"Invalid GIST layers; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for layer_name in GIST_LAYER_PREFIXES:
        if layer_name not in selection:
            raise ValueError(f"Missing GIST layer: {layer_name}")
        top_k = selection_count(len(selection[layer_name]), selection_ratio)
        selected_count = sum(bool(value) for value in selection[layer_name].values())
        if selected_count != top_k:
            raise ValueError(
                f"Invalid GIST selection for {layer_name}: "
                f"selected {selected_count}, expected {top_k}"
            )
