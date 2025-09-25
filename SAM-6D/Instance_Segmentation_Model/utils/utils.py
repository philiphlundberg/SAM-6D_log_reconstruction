def load_exact_section(settings_file: str, section: str, cli_overrides: dict):
    from omegaconf import OmegaConf
    cfg_all = OmegaConf.load(settings_file)
    if section not in cfg_all:
        raise KeyError(f"Section '{section}' not found in {settings_file}")
    cfg = cfg_all[section]
    ignore = {"settings_file", "section"}
    overrides = {k: v for k, v in cli_overrides.items() if k not in ignore and v is not None}
    return OmegaConf.merge(cfg, overrides)
