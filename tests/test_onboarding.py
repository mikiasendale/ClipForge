"""Acceptance §11.2 — enforce exact pick counts; §11.8 — config reflected."""
from fastapi.testclient import TestClient
from app import main as m


def _client(home):
    m.cfg.DB_PATH = home / "data" / "clipforge.db"
    return TestClient(m.app)


def test_onboarding_wrong_counts_returns_400(home):
    client = _client(home)
    body = {"selected": {"football": [{"platform_channel_id": "UC1", "title": "x"}]}}  # need 3
    r = client.post("/onboarding", json=body)
    assert r.status_code == 400
    assert "football" in r.json()["detail"]


def test_onboarding_correct_counts_saves(home):
    client = _client(home)
    sel = {
        "football": [{"platform_channel_id": f"UCf{i}", "title": f"F{i}", "subs": 1000} for i in range(3)],
        "boxing": [{"platform_channel_id": f"UCb{i}", "title": f"B{i}"} for i in range(3)],
        "cats_silent": [{"platform_channel_id": f"UCs{i}", "title": f"S{i}"} for i in range(2)],
        "cats_compilations": [{"platform_channel_id": f"UCc{i}", "title": f"C{i}"} for i in range(2)],
    }
    r = client.post("/onboarding", json={"selected": sel})
    assert r.status_code == 200
    assert r.json()["saved"] == 10
    assert m.db.channel_count() == 10


def test_config_model_edit_reflected_without_code(home, monkeypatch):
    # §11.8: writing config.yaml and reloading is reflected in get_config
    cfg_file = home / "config.yaml"
    import yaml
    base = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    base["openrouter"]["models"]["text"] = "test/model-xyz"
    cfg_file.write_text(yaml.safe_dump(base), encoding="utf-8")
    monkeypatch.setattr(m.cfg, "CONFIG_PATH", cfg_file)
    c = m.cfg.load_config(cfg_file)
    assert c.get("openrouter.models.text") == "test/model-xyz"
