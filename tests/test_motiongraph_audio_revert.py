"""Reverting a renamed species' sound events back to the donor's stock names.

A mod's custom audio only plays with the whole chain installed - the proxy dll,
the loader mod, the raw banks and the registry fragment in <Mod>/Audio/. Publishing
a copy without that chain means putting the donor's names back in the motiongraph,
so the engine resolves them out of the game's own audio OVLs again.

The revert must therefore depend on NONE of that chain: not the mod's bank (it may
already be deleted - that is usually why someone wants this) and not an
``_events.ovl``, which a mod never ships in the first place.
"""

import struct
from pathlib import Path

import pytest

from source.formats.motiongraph import audio_events
from source.formats.motiongraph.audio_events import (classify_revert, event_ids_in_bank,
                                                     fnv1_32, find_mod_events_bank,
                                                     revert_plan, stock_event_ids)
from source.formats.motiongraph.staging import copy_motiongraph_family, family_hashes
from source.formats.motiongraph.static_patch import apply_string_renames


FIXTURE = Path(
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game"
    r"\Dinosaur Files\Motiongraph Research\Appended Clip String Test X"
    r"\Source Test W\Acrocanthosaurus_Female.ovl"
)


def build_bank(event_ids, *, tag=b"HIRC"):
    """A minimal Wwise bank: BKHD then a HIRC holding one EVENT per id."""
    objects = b""
    for value in event_ids:
        payload = struct.pack("<I", value) + b"\x00\x00"
        objects += bytes([audio_events.EVENT]) + struct.pack("<I", len(payload)) + payload
    hirc = struct.pack("<I", len(event_ids)) + objects
    return (b"BKHD" + struct.pack("<I", 8) + b"\x01\x00\x00\x00\x02\x00\x00\x00"
            + tag + struct.pack("<I", len(hirc)) + hirc)


def test_a_mods_loose_bnk_is_a_real_bank_not_an_ovl_stub(tmp_path):
    """Mods ship <Mod>/Audio/<prefix>_events.bnk, never a packed _events.ovl."""
    ids = {fnv1_32("Ultima_AttackRoar"), fnv1_32("Ultima_Snort")}
    bank = tmp_path / "Audio" / "ultima_events.bnk"
    bank.parent.mkdir()
    bank.write_bytes(build_bank(sorted(ids)))

    assert event_ids_in_bank(bank) == ids
    # located from the mod root and from the Audio folder, case-insensitively
    assert find_mod_events_bank(tmp_path, "Ultima") == bank
    assert find_mod_events_bank(tmp_path / "Audio", "ULTIMA") == bank
    assert find_mod_events_bank(tmp_path, "SomeoneElse") is None


def test_revert_targets_are_verified_against_the_stock_registry(monkeypatch, tmp_path):
    """Restorable means the STOCK game declares the target, shared banks included."""
    registry = (
        '<evententry stop_start_fnv="%d" event_fnv="%d" />'
        '<evententry start_fnv="%d" />'
        % (fnv1_32("IndominusRex_AttackRoar"), fnv1_32("IndominusRex_AttackRoar"),
           fnv1_32("IndominusRex_SocialCall"))
    )
    monkeypatch.setattr(audio_events, "read_stock_registry",
                        lambda game_root, game: registry)
    ids = stock_event_ids(tmp_path, "Jurassic World Evolution 3")

    names = ["UltimasaurusCE_AttackRoar", "UltimasaurusCE_SocialCall",
             "UltimasaurusCE_ChaosScreech", "Unrelated_Thing"]
    rows = classify_revert(names, "UltimasaurusCE", "IndominusRex", ids)

    assert rows == [
        ("UltimasaurusCE_AttackRoar", "IndominusRex_AttackRoar", True),
        # a SHARED-bank sound is a legitimate revert target, not a refusal
        ("UltimasaurusCE_SocialCall", "IndominusRex_SocialCall", True),
        # invented by the modder: no stock event to fall back to
        ("UltimasaurusCE_ChaosScreech", "IndominusRex_ChaosScreech", False),
    ]
    # a name that is not this species' at all is never touched
    assert all(name != "Unrelated_Thing" for name, _target, _ok in rows)


def test_revert_without_a_game_folder_verifies_nothing_and_says_so():
    """No game folder is a real case - the check is skipped, not faked."""
    rows = classify_revert(["Ultima_Roar", "Ultima_Invented"], "Ultima", "Indominus", None)
    assert [ok for _name, _target, ok in rows] == [True, True]


def test_revert_plan_mirrors_rename_plan_longest_first():
    names = ["Ultima_A", "Ultima_LongerName", "Ultima_Mid"]
    plan = revert_plan(names, "Ultima", "Indominus")
    assert plan == [
        ("Ultima_LongerName", "Indominus_LongerName"),
        ("Ultima_Mid", "Indominus_Mid"),
        ("Ultima_A", "Indominus_A"),
    ]
    assert [len(old) for old, _new in plan] == sorted(
        (len(old) for old, _new in plan), reverse=True)
    assert revert_plan(["Other_Roar"], "Ultima", "Indominus") == []


def test_registry_read_is_cached_per_file_identity(monkeypatch, tmp_path):
    """Unpacking audiometadata.ovl costs seconds; the fragment builder and the
    revert classifier must not each pay it."""
    meta = tmp_path / "Win64" / "ovldata" / "Content0" / "Audio" / "MetaData"
    meta.mkdir(parents=True)
    (meta / "audiometadata.ovl").write_bytes(b"not really an ovl")
    calls = []

    def fake_open(path, game):
        calls.append(path)
        raise RuntimeError("stop here; the cache is what is under test")

    monkeypatch.setattr(audio_events, "_open_ovl", fake_open)
    audio_events._REGISTRY_CACHE.clear()
    with pytest.raises(RuntimeError):
        audio_events.read_stock_registry(tmp_path, "Jurassic World Evolution 3")
    assert len(calls) == 1

    audio_events._REGISTRY_CACHE[(
        str((meta / "audiometadata.ovl").resolve()),
        (meta / "audiometadata.ovl").stat().st_size,
        int((meta / "audiometadata.ovl").stat().st_mtime),
    )] = "<cached/>"
    assert audio_events.read_stock_registry(
        tmp_path, "Jurassic World Evolution 3") == "<cached/>"
    assert len(calls) == 1


def test_rename_then_revert_restores_every_stock_name(tmp_path):
    """End to end on a pinned family: rename to a LONGER prefix, then revert.

    Longer on purpose - that forces the relocation path on the way out, so the
    revert has to find names that live in an appended pool tail rather than where
    the vanilla file put them.
    """
    if not FIXTURE.is_file():
        pytest.skip("Pinned Acrocanthosaurus fixture is unavailable")
    donor, prefix = "Acrocanthosaurus", "AcrocanthosaurusCE"
    copy_motiongraph_family(FIXTURE, tmp_path / "source")
    source = tmp_path / "source" / FIXTURE.name
    pristine = family_hashes(source)

    names = audio_events.scan_graph_event_names(
        source, donor, "Jurassic World Evolution 3")[:2]
    assert names, "fixture graph fires no donor-prefixed sounds"

    copy_motiongraph_family(source, tmp_path / "renamed")
    renamed = tmp_path / "renamed" / FIXTURE.name
    forward = [(name, prefix + name[len(donor):]) for name in names]
    apply_string_renames(source, renamed, forward)
    assert set(audio_events.scan_graph_event_names(
        renamed, prefix, "Jurassic World Evolution 3")) == {new for _old, new in forward}

    # Now the actual subject: put them back, with no bank and no game folder.
    scanned = audio_events.scan_graph_event_names(
        renamed, prefix, "Jurassic World Evolution 3")
    plan = revert_plan(scanned, prefix, donor)
    copy_motiongraph_family(renamed, tmp_path / "reverted")
    reverted = tmp_path / "reverted" / FIXTURE.name
    apply_string_renames(renamed, reverted, plan)

    restored = audio_events.scan_graph_event_names(
        reverted, donor, "Jurassic World Evolution 3")
    assert set(names).issubset(restored)
    assert audio_events.scan_graph_event_names(
        reverted, prefix, "Jurassic World Evolution 3") == []
    # the source family is never written to, and companions are never rewritten
    assert family_hashes(source) == pristine
    for stage in (renamed, reverted):
        assert {name: digest for name, digest in family_hashes(stage).items()
                if name != stage.name.lower()} == {
            name: digest for name, digest in pristine.items()
            if name != source.name.lower()}
