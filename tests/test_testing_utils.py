from areal.utils import testing_utils


def test_lazy_model_paths_resolve_only_requested_key(monkeypatch):
    calls: list[tuple[str, str]] = []

    def resolve(local_path: str, hf_id: str) -> str:
        calls.append((local_path, hf_id))
        return f"resolved/{hf_id}"

    monkeypatch.setattr(testing_utils, "get_model_path", resolve)
    paths = testing_utils._LazyModelPaths(
        {
            "first": ("local/first", "org/first"),
            "second": ("local/second", "org/second"),
        }
    )

    assert list(paths) == ["first", "second"]
    assert len(paths) == 2
    assert calls == []

    assert paths["second"] == "resolved/org/second"
    assert paths["second"] == "resolved/org/second"
    assert calls == [("local/second", "org/second")]
