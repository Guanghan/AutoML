def test_config():
    from src.utils.read_configure import Config
    config = Config("../configs/example.yaml")
    print(config)
    assert isinstance(config, Config)


def test_desc2config():
    from src.utils.read_configure import desc2config
    assert False
