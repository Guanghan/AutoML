def test_config():
    from src.utils.utils_cfg import Config
    config = Config("../configs/example.yaml")
    print(config)
    assert isinstance(config, Config)


def test_desc2config():
    from src.utils.utils_cfg import desc2config
    assert False
