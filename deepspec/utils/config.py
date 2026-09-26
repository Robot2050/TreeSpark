import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
import torch
import yaml


class ConfigNode(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return ConfigNode(super().copy())


def to_config_node(value):
    if isinstance(value, ConfigNode):
        return ConfigNode({k: to_config_node(v) for k, v in value.items()})
    if isinstance(value, dict):
        return ConfigNode({k: to_config_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_config_node(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_config_node(v) for v in value)
    return value


def config_to_plain_dict(value):
    if isinstance(value, ConfigNode):
        return {k: config_to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: config_to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [config_to_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return [config_to_plain_dict(v) for v in value]
    return value


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ConfigNode):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, FunctionType):
            return f"<function {obj.__name__}>"
        if isinstance(obj, type):
            return f"<class '{obj.__name__}'>"
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, ConfigNode):
            return config_to_plain_dict(obj)
        if isinstance(obj, (Namespace, SimpleNamespace)):
            return vars(obj)
        return super().default(obj)


def _parse_scalar(value):
    return yaml.safe_load(value)


def load_config(path):
    path = os.path.abspath(path)
    module_name = Path(path).stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {}
    for name in dir(module):
        if name.startswith("__"):
            continue
        value = getattr(module, name)
        if isinstance(value, ModuleType):
            continue
        config[name] = value
    return to_config_node(config)


def finalize_config(cfg):
    cfg = to_config_node(cfg)
    if not isinstance(cfg, dict):
        return cfg
    finalize_cfg = cfg.get("finalize_cfg")
    if callable(finalize_cfg):
        new_cfg = finalize_cfg(cfg)
        if new_cfg is not None:
            cfg = new_cfg
    return to_config_node(cfg)


def parse_opts_to_config(opts, cfg):
    if not opts:
        return finalize_config(cfg)
    for opt in opts:
        name, value = opt.split("=", 1)
        parts = name.split(".")
        current = cfg
        for part in parts[:-1]:
            if part not in current:
                raise KeyError(f"Unknown config key in --opts: {name}")
            if not isinstance(current[part], dict):
                raise TypeError(
                    f"Cannot set nested config key {name}: {part} is not a mapping."
                )
            current = current[part]
        if parts[-1] not in current:
            raise KeyError(f"Unknown config key in --opts: {name}")
        current[parts[-1]] = to_config_node(_parse_scalar(value))
    return finalize_config(cfg)
