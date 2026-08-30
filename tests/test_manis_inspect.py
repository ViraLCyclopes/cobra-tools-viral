from types import SimpleNamespace

from source.formats.manis.inspect import (
    ERROR,
    INFO,
    WARNING,
    Issue,
    clip_report,
    transform_blob_indices,
    verify_bundle,
    worst_severity,
)


QVVF = 12
SCALAR = 0


def mani_info(name, *, compression=0, frames=10, pos=3, ori=4, scl=0, floats=2,
              target=8, duration=1.0):
    return SimpleNamespace(
        name=name,
        dtype=SimpleNamespace(compression=compression),
        duration=duration,
        frame_count=frames,
        pos_bone_count=pos,
        ori_bone_count=ori,
        scl_bone_count=scl,
        float_count=floats,
        target_bone_count=target,
    )


def bundle(*infos, mani_count=None, names=None):
    return SimpleNamespace(
        name="test.manis",
        mani_infos=list(infos),
        mani_count=len(infos) if mani_count is None else mani_count,
        names=[i.name for i in infos] if names is None else names,
        eoh=None,
    )


def test_transform_blob_indices_skips_interleaved_scalar_blobs():
    """Clip N's transform blob is the Nth QVVF blob, never blob 2*N."""
    blobs = [
        {"track_type": QVVF}, {"track_type": SCALAR},
        {"track_type": QVVF},
        {"track_type": QVVF}, {"track_type": SCALAR},
    ]

    assert transform_blob_indices(blobs) == [0, 2, 3]


def test_clip_report_marks_uncompressed_clips_as_out_of_scope_for_acl():
    manis = bundle(mani_info("jwe2$walk"))

    report = clip_report(manis, 0, data=None)

    assert report.is_acl is False
    assert report.compressed is False
    assert "ACL actions do not apply" in report.notes[0]


def test_clip_report_flags_a_compressed_clip_with_no_locatable_blob():
    manis = bundle(mani_info("jwe3$walk", compression=1))

    report = clip_report(manis, 0, data=b"")

    assert report.is_acl is False
    assert "no ACL blob" in report.notes[0]


def test_verify_accepts_an_ordinary_uncompressed_bundle():
    manis = bundle(mani_info("a$walk"), mani_info("a$run"))

    assert verify_bundle(manis) == []


def test_verify_rejects_duplicate_and_empty_clip_names():
    manis = bundle(mani_info("a$walk"), mani_info("a$walk"), mani_info("  "))

    issues = verify_bundle(manis)

    assert worst_severity(issues) == ERROR
    assert any("duplicate clip name" in i.message for i in issues)
    assert any("empty name" in i.message for i in issues)


def test_verify_catches_a_stale_clip_count_after_adding_a_clip():
    """The failure mode when a bundle gains a clip but the header is not updated."""
    manis = bundle(mani_info("a$walk"), mani_info("a$preen2"), mani_count=1)

    issues = verify_bundle(manis)

    assert any("mani_count is 1" in i.message for i in issues)
    assert worst_severity(issues) == ERROR


def test_verify_catches_a_name_list_that_did_not_grow_with_the_clips():
    manis = bundle(mani_info("a$walk"), mani_info("a$preen2"), names=["a$walk"])

    issues = verify_bundle(manis)

    assert any("name list holds 1 entries" in i.message for i in issues)


def test_verify_rejects_a_channel_count_above_the_target_skeleton():
    manis = bundle(mani_info("a$walk", ori=99, target=8))

    issues = verify_bundle(manis)

    assert any("ori_bone_count is 99" in i.message for i in issues)


def test_verify_treats_trailing_bytes_as_information_not_failure():
    """JWE3 appends ACL bulk past the header region; that is normal, not a fault."""
    manis = bundle(mani_info("a$walk"))
    manis.eoh = 100

    issues = verify_bundle(manis, data=None, file_size=400)

    assert worst_severity(issues) == INFO
    assert any("300 bytes after the header region" in i.message for i in issues)


def test_verify_rejects_a_header_that_runs_past_the_file():
    manis = bundle(mani_info("a$walk"))
    manis.eoh = 900

    issues = verify_bundle(manis, data=None, file_size=400)

    assert worst_severity(issues) == ERROR
    assert any("runs past the file" in i.message for i in issues)


def test_issue_renders_with_its_clip_name():
    assert str(Issue(WARNING, "frame_count is zero", "a$walk")) == (
        "WARNING [a$walk]: frame_count is zero")
