from cybsage_marl.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.project_name == "CybSAGE-MARL"
    assert cfg.drive_output_folder_id == "1Oh9o3ipRfNZmeyOl2uatBSHdGrnvnfUc"
