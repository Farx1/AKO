from pathlib import Path

from scripts.summarize_reports import build_markdown

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "rtx3070"


def test_decoder_profile_has_config_and_json_values():
    md = build_markdown(REPORTS)
    assert "| Layers | 4 | `reports/rtx3070/model_profile.json` `model.n_layers` |" in md
    assert "KV cache holding 511 tokens" in md
    assert "| prefill | 11.692 | 11.158 | 9.038 | 1.29× | 1.24× (torch.compile) |" in md
    assert "| decode | 10.057 | 7.945 | 12.228 | 0.82× | 0.65× (torch.compile) |" in md
