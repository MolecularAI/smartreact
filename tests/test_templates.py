from __future__ import annotations

import pytest

from smartreact import load_templates
from smartreact.types import ReactionTemplate


class TestLoadTemplates:
    def test_all_reactions(self):
        templates = load_templates(reaction_list="all")
        assert len(templates) > 0
        names = {t.name for t in templates}
        assert "suzuki" in names

    def test_all_loads_more_than_subset(self):
        all_templates = load_templates(reaction_list="all")
        subset = load_templates(reaction_list=["suzuki"])
        assert len(all_templates) > len(subset)

    def test_single_reaction(self):
        templates = load_templates(reaction_list=["suzuki"])
        assert all(t.name == "suzuki" for t in templates)
        assert len(templates) > 0

    def test_template_fields(self):
        templates = load_templates(reaction_list=["suzuki"])
        t = templates[0]
        assert isinstance(t, ReactionTemplate)
        assert t.smarts
        assert t.reactant_categories
        assert t.name == "suzuki"

    def test_empty_reaction_list_raises(self):
        with pytest.raises(ValueError, match="No usable reaction templates"):
            load_templates(reaction_list=["nonexistent_reaction"])

    def test_string_reaction_list_raises(self):
        with pytest.raises(ValueError, match="reaction_list must be"):
            load_templates(reaction_list="suzuki")  # type: ignore[arg-type]


class TestLoadTemplatesMalformed:
    """load_templates row- and header-level validation."""

    @staticmethod
    def _patch_data(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        from unittest.mock import MagicMock

        from smartreact import templates as templates_mod

        file_handle = MagicMock()
        file_handle.read_text.return_value = text
        data_root = MagicMock()
        data_root.__truediv__.return_value = file_handle
        monkeypatch.setattr(templates_mod, "_DATA", data_root)

    def test_missing_required_header_column_raises(self, monkeypatch):
        # Drop the reactant_categories column entirely.
        text = "reaction_name\ttemplate_id\treaction_template\nsuzuki\t1\t[c:1][Br:2]>>[c:1]\n"
        self._patch_data(monkeypatch, text)
        with pytest.raises(ValueError, match="must contain columns"):
            load_templates(reaction_list="all")

    def test_row_without_dot_in_reactant_categories_is_skipped(self, monkeypatch, caplog):
        import logging

        text = (
            "reaction_name\ttemplate_id\treaction_template\treactant_categories\n"
            "bad_row\t1\t[c:1][Br:2]>>[c:1]\tNoDotHere\n"
            "good_row\t2\t[c:1][Br:2]>>[c:1]\tA.B\n"
        )
        self._patch_data(monkeypatch, text)
        with caplog.at_level(logging.ERROR, logger="smartreact.templates"):
            templates = load_templates(reaction_list="all")

        names = {t.name for t in templates}
        assert "bad_row" not in names
        assert "good_row" in names
        assert any("no '.' separator" in msg for msg in caplog.messages)
