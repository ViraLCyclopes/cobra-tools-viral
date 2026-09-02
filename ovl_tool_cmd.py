"""
ovl_tool_cmd.py

Command-line OVL tool using the same OvlFile API as ovl_tool_gui.

Subcommands:
  new      - create a new OVL from a folder
  extract  - extract files from an OVL
  inject   - inject/replace files into an OVL
  retarget-family - fixed-width, hash-preserving JWE3 species-family retarget
  motiongraph-fields - inspect fields and optionally create a fixed-size patch plan
  motiongraph-apply  - apply a verified patch plan without rebuilding topology
  motiongraph-report - render the anonymous state graph or decision tree
  motiongraph-capacity - audit reusable allocation slack and dormant topology
  motiongraph-census   - count decoded graph objects; diff two builds for orphans
  motiongraph-chooser  - list/grow/re-weight a random animation chooser
  motiongraph-diff     - capture one build's edits as a replayable patch plan
  motiongraph-grow-null-prefix - experimental fixed-pool logical array growth
  motiongraph-retarget-clip - repoint references to an existing same-pool clip string
  motiongraph-string-slot   - replace text within one existing string allocation

Examples:
  ovl_tool_cmd.py extract -i path/to/main.ovl 
  ovl_tool_cmd.py new -i this/folder/ -g "Planet Zoo" -o Main.ovl
  ovl_tool_cmd.py inject -f test/test.lua  -g "Jurassic World Evolution 3" --in-place path/to/main.ovl

"""
from __future__ import annotations

from utils import config
from utils.logs import get_global_listener, logging_setup # type: ignore
import logging
# Command output belongs on stdout.  Avoid creating/rotating a log beside the
# tool, which also makes parallel CLI runs contend for the same file.
logging_setup("ovl_tool_cmd", log_to_file=False)

import argparse
import json
import os
import sys
from typing import Iterable, List, Optional

# -----------------------------------------------------------------------------
# Bootstrapping: repo root, shared formats, logging shim
# -----------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import contextlib
from modules.formats.shared import DummyReporter 

class BuildReporter(DummyReporter):
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.error_files = []

    def show_warning(self, msg: str):
        self.warnings.append(msg)

    def show_error(self, exception: Exception):
        self.errors.append(exception)

    def iter_progress(self, iterable, message, cond=True):
        for item in iterable:
            yield item

    @contextlib.contextmanager
    def report_error_files(self, operation):
        yield self.error_files
  

from generated.formats.ovl import games, OvlFile
from generated.formats.ovl_base.enums.Compression import Compression
from utils.config import Config

# In the GUI, logging.success is provided by their logging wrapper; here we alias
if not hasattr(logging, "success"):
    def success(msg, *args, **kwargs):
        logging.getLogger("cobra-tools").info(msg, *args, **kwargs)
    logging.success = success  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_exists(path: str, kind: str = "file") -> None:
    if kind == "file" and not os.path.isfile(path):
        die(f"{kind.capitalize()} does not exist: {path}")
    if kind == "dir" and not os.path.isdir(path):
        die(f"{kind.capitalize()} does not exist: {path}")


def find_common_root(files: Iterable[str]) -> str:
    paths = [os.path.abspath(p) for p in files]
    if not paths:
        return ""
    if len(paths) == 1:
        return os.path.dirname(paths[0])
    return os.path.commonpath(paths)


def resolve_game_label(label: Optional[str]) -> Optional[str]:
    """
    Convert a user-facing game label to the same value the GUI uses.

    The combo box in the GUI is built from [g.value for g in games], and
    game_changed() sets ovl_data.game to that value, so we mirror that.
    """
    if not label:
        return None
    for g in games:
        if label == getattr(g, "value", None) or label == getattr(g, "name", None):
            return g.value
    return label


def game_choices() -> List[str]:
    try:
        return [g.value for g in games]
    except Exception:
        return []


def compression_choices() -> List[str]:
    return [c.name for c in Compression]


# -----------------------------------------------------------------------------
# Command line operations 
# -----------------------------------------------------------------------------

def cmd_new(args: argparse.Namespace) -> None:
    """
    File > New (from folder): create an OVL from a directory and save it.

    Mirrors MainWindow.create_ovl + save in ovl_tool_gui:
        self.ovl_data.clear()
        self.game_changed()
        self.ovl_data.create(ovl_dir)
        self.ovl_data.save(filepath, commands={"update_aux": cfg["update_aux"]})
    """
    in_dir = os.path.abspath(args.input)
    out_ovl = os.path.abspath(args.output)

    ensure_exists(in_dir, "dir")
    if os.path.exists(out_ovl) and not args.force:
        die(f"Output file already exists: {out_ovl} (use --force to overwrite)")

    game = resolve_game_label(args.game)
    if not game:
        die("You must specify --game for 'new'.")

    ovl = OvlFile()
    config = Config(REPO_ROOT)
    config.load()
    ovl.cfg = config
    ovl.game = game  # same as game_changed() ultimately does
    ovl.load_hash_table()

    if args.compression:
        try:
            ovl.user_version.compression = Compression[args.compression]
        except KeyError:
            die(f"Unknown compression '{args.compression}'. Valid: {', '.join(compression_choices())}")

    logging.info("Creating OVL from %s", in_dir)
    logging.info("Game: %s", ovl.game)

    try:
        ovl.clear()    
        ovl.create(in_dir)
    except Exception as e:
        die(f"OvlFile.create failed: {e!r}")

    commands = {"update_aux": args.update_aux}
    try:
        ovl.save(out_ovl, commands=commands)
    except Exception as e:
        die(f"OvlFile.save failed: {e!r}")

    logging.success("Created OVL: %s", out_ovl)


def cmd_extract(args: argparse.Namespace) -> None:
    """
    Extract from an OVL.

    GUI equivalents:
      - MainWindow._extract_all -> ovl.extract(out_dir, only_types=only_types)
      - drag_files -> ovl.extract(temp_dir, only_names=file_names)
    """
    ovl_path = os.path.abspath(args.ovl)
    ensure_exists(ovl_path, "file")

    if args.output:
        out_dir = os.path.abspath(args.output)
    else:
        # Default: <same_dir>/<ovl_basename_without_ext>
        ovl_dir = os.path.dirname(ovl_path)
        ovl_name = os.path.splitext(os.path.basename(ovl_path))[0]
        out_dir = os.path.join(ovl_dir, ovl_name)

    os.makedirs(out_dir, exist_ok=True)

    ovl = OvlFile()

    # Optional override: if user passes -g, we force that game
    game = resolve_game_label(args.game)
    commands = {}
    if game:
        commands["game"] = game
        logging.info("Using game preset: %s", game)
    else:
        logging.info("No game preset supplied; OvlFile will auto-detect.")
    logging.info("Loading archive %s", ovl_path)

    try:
        ovl.load(ovl_path, commands)
        logging.info("Detected game from archive: %s", getattr(ovl, "game", "<unknown>"))
    except Exception as e:
        die(f"OvlFile.load failed: {e!r}")

    only_types = args.type or None
    only_names = args.name or None

    logging.info("Extracting to %s", out_dir)
    if only_types:
        logging.info("Only types: %s", ", ".join(only_types))
    if only_names:
        logging.info("Only names: %s", ", ".join(only_names))

    kwargs = {}
    if only_types:
        kwargs["only_types"] = only_types
    if only_names:
        kwargs["only_names"] = only_names

    try:
        ovl.extract(out_dir, **kwargs)
    except Exception as e:
        die(f"OvlFile.extract failed: {e!r}")

    logging.success("Extracted OVL to %s", out_dir)


def cmd_inject(args: argparse.Namespace) -> None:
    """
    Inject files into an OVL, using OvlFile.add_files(files, common_root_dir)
    (same pattern as MainWindow.inject_files in the GUI).
    """
    ovl_src = os.path.abspath(args.ovl)
    ensure_exists(ovl_src, "file")

    game = resolve_game_label(args.game)
    commands = {}
    if game:
        commands["game"] = game
        logging.info("Using game preset: %s", game)
    else:
        logging.info("No game preset supplied; OvlFile will auto-detect.")

    ovl = OvlFile()
    ovl.game = game
    ovl.load_hash_table()

    logging.info("Loading archive %s", ovl_src)

    try:
        ovl.clear()
        ovl.load(ovl_src, commands)
    except Exception as e:
        die(f"OvlFile.load failed: {e!r}")

    try:
        logging.info("Detected game from archive: %s", getattr(ovl, "game", "<unknown>"))
    except Exception:
        pass


    # Collect files to inject
    files_to_inject: List[str] = []

    if args.input:
        in_dir = os.path.abspath(args.input)
        ensure_exists(in_dir, "dir")
        for root, _, files in os.walk(in_dir):
            for name in files:
                files_to_inject.append(os.path.join(root, name))

    if args.file:
        for p in args.file:
            files_to_inject.append(os.path.abspath(p))

    if not files_to_inject:
        die("No files to inject (use --input folder and/or --file path).")

    files_to_inject = sorted(set(files_to_inject))

    # Choose a common root like the GUI does for relative names
    if args.input:
        common_root = os.path.abspath(args.input)
    else:
        common_root = find_common_root(files_to_inject)
        if not common_root:
            die("Could not determine common root directory for injected files.")

    logging.info("Injecting %d files (root: %s)", len(files_to_inject), common_root)

    if args.update:
        # Write into the loaders that are already there instead of removing and
        # re-creating them. A re-created loader allocates fresh pools and data entries,
        # which leaves the OVL structurally different from the one that was loaded; JWE3
        # rejects that even when the injected file is byte-identical to what came out,
        # while a plain load/save of the same OVL loads fine.
        for file_path in files_to_inject:
            name = os.path.basename(file_path)
            loader = ovl.loaders.get(name)
            if loader is None:
                die(f"--update needs the file to already exist in the OVL: {name}")
            try:
                loader.update_in_place(file_path)
            except NotImplementedError:
                die(f"--update is not supported for {name} ({type(loader).__name__})")
            except Exception as e:
                die(f"update_in_place failed for {name}: {e}")
            logging.success("Updated %s in place", name)
    else:
        try:
            ovl.add_files(files_to_inject, common_root)
        except Exception as e:
            die(f"OvlFile.add_files failed: {e!r}")

    # Decide output path
    if args.in_place:
        out_ovl = ovl_src
    else:
        if not args.output:
            die("You must specify --output when not using --in-place.")
        out_ovl = os.path.abspath(args.output)

    commands_save = {"update_aux": args.update_aux}

    logging.info("Saving archive to %s", out_ovl)
    try:
        ovl.save(out_ovl, commands=commands_save)
    except Exception as e:
        die(f"OvlFile.save failed: {e!r}")

    logging.success("Injected files into %s", out_ovl)


def cmd_retarget_family(args: argparse.Namespace) -> None:
    """Build a complete JWE3 family using a fixed-width djb2-collision alias."""
    from source.formats.ovl.retarget import retarget_family

    try:
        report = retarget_family(
            args.ovl,
            args.output,
            args.donor,
            args.alias,
            game=resolve_game_label(args.game),
            force=args.force,
        )
    except Exception as exc:
        die(f"retarget-family failed: {exc}")

    logging.success(
        "Retargeted %d loaders and %d STATIC pool strings (%d bytes) in pools %s",
        report.renamed_loaders,
        report.pool_strings,
        report.pool_bytes,
        list(report.pool_indices),
    )
    logging.success(
        "STATIC compressed size: %d -> %d",
        report.static_compressed_before,
        report.static_compressed_after,
    )
    logging.success("Verified and wrote %d family files:", len(report.output_files))
    for path in report.output_files:
        logging.success("  %s", path)


def cmd_motiongraph_fields(args: argparse.Namespace) -> None:
    """Locate verified activity fields and optionally emit a patch plan."""
    from pathlib import Path
    from source.formats.motiongraph.edit import build_patch_plan, locate_fields, save_plan

    source = Path(args.ovl).resolve()
    ensure_exists(str(source), "file")
    try:
        rows, mismatches = locate_fields(
            source,
            name=args.name,
            game=resolve_game_label(args.game),
            activity=args.activity,
            activity_type=args.activity_type,
            field=args.field,
        )
    except Exception as exc:
        die(f"motiongraph field scan failed: {exc}")

    bad = [field for row in rows for field in row["fields"] if not field["verified"]]
    total = sum(len(row["fields"]) for row in rows)
    for row in rows:
        print(
            f"\n{row['activity']} [{row['activity_type']}] "
            f"pool {row['pool']} @ {row['payload_offset']}"
        )
        for item in row["fields"]:
            display = item.get("curve_value", item["value"])
            verified = "" if item["verified"] else " MISMATCH"
            print(
                f"  {item['path']:<38} {item['kind']:<8} "
                f"pool {item['pool']}:{item['offset']} = {display}{verified}"
            )
    print(
        f"\nactivities={len(rows)} fields={total} "
        f"verified={total - len(bad)} mismatched={len(bad)}"
    )
    for struct_name, (declared, actual) in sorted(mismatches.items()):
        print(f"skipped {struct_name} array: declared stride {declared}, actual {actual}")
    if args.json:
        json_path = Path(args.json).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logging.success("Wrote field report: %s", json_path)
    if bad:
        die("At least one computed address disagrees with the decoded value")

    writes = [args.set_value is not None, bool(args.set_flag), args.set_enum is not None]
    if any(writes):
        if not args.field or not args.plan:
            die("A write selection requires both --field and --plan")
        try:
            plan = build_patch_plan(
                rows, source, args.name, args.field,
                value=args.set_value,
                flag_ops=args.set_flag or None,
                enum_name=args.set_enum,
            )
            plan_path = Path(args.plan).resolve()
            save_plan(plan_path, plan)
        except Exception as exc:
            die(f"Could not create patch plan: {exc}")
        logging.success("Wrote %d-edit patch plan: %s", len(plan["edits"]), plan_path)


def cmd_motiongraph_apply(args: argparse.Namespace) -> None:
    """Apply a same-width motiongraph plan to a staged OVL."""
    from pathlib import Path
    from source.formats.motiongraph.edit import apply_patch_plan

    source = Path(args.ovl).resolve()
    output = Path(args.output).resolve()
    plan_path = Path(args.plan).resolve()
    ensure_exists(str(source), "file")
    ensure_exists(str(plan_path), "file")
    if output.exists() and not args.force:
        die(f"Output already exists: {output} (use --force to replace it)")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        report = apply_patch_plan(
            source, output, plan, game=resolve_game_label(args.game)
        )
    except Exception as exc:
        die(f"motiongraph patch failed: {exc}")
    logging.success(
        "Patched %d addresses across %d STATIC pools (%d changed bytes)",
        report.edits, report.pools, report.changed_bytes,
    )
    logging.success(
        "STATIC topology preserved: %d pools, %d fragments, %d uncompressed bytes",
        report.static_pools, report.static_fragments, report.uncompressed_size,
    )
    logging.success(
        "STATIC compressed size: %d -> %d", report.compressed_before,
        report.compressed_after,
    )
    logging.success("Wrote staged OVL: %s", report.output)


def cmd_motiongraph_report(args: argparse.Namespace) -> None:
    """Render a state/activity report or the MRF decision tree."""
    from pathlib import Path
    from source.formats.motiongraph.edit import load_motiongraph
    from source.formats.motiongraph.report import build_decision_report, build_state_report

    source = Path(args.ovl).resolve()
    output = Path(args.output).resolve()
    ensure_exists(str(source), "file")
    if args.kind != "state" and args.json:
        die("--json is only available for the state report")
    try:
        _, loader = load_motiongraph(source, args.name, resolve_game_label(args.game))
        if args.kind == "state":
            report, payload, stats = build_state_report(loader)
        else:
            report, stats = build_decision_report(loader)
            payload = None
    except Exception as exc:
        die(f"motiongraph report failed: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logging.success("Wrote %s report: %s", args.kind, output)
    if args.json:
        json_path = Path(args.json).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.success("Wrote state JSON: %s", json_path)
    if args.kind == "state":
        print(f"states={stats.states} edges={stats.edges} conditions={stats.conditions}")
    else:
        print(
            f"states={stats.states} decision_nodes={stats.decision_nodes} "
            f"blocks={stats.decision_blocks} opcodes={stats.opcodes}"
        )


def cmd_motiongraph_diff(args: argparse.Namespace) -> None:
    """Diff two builds' STATIC pools into a replayable patch plan.

    This is the recovery route for motiongraph edits. An extracted .motiongraph is
    faithful XML but SEMANTIC - it carries pool_type, never pool:offset - and
    MotiongraphLoader.create() is not implemented, so there is no way back in from
    it. Recording what changed as bytes and replaying with motiongraph-apply is.

    Fragment retargets and topology growth cannot be expressed as a byte patch;
    both are reported so a recovery is never silently partial.
    """
    from pathlib import Path
    from source.formats.motiongraph.diff import diff_motiongraph, write_plan

    game = resolve_game_label(args.game)
    stock, edited = Path(args.ovl).resolve(), Path(args.against).resolve()
    ensure_exists(str(stock), "file")
    ensure_exists(str(edited), "file")
    try:
        report = diff_motiongraph(stock, edited, game)
    except Exception as exc:
        die(f"motiongraph diff failed: {exc}")
    for warning in report.warnings:
        logging.warning(warning)
    print(f"changed_pools={report.changed_pools} changed_bytes={report.changed_bytes} "
          f"edits={report.edits} fragments_retargeted={report.fragments_retargeted} "
          f"fragment_delta={report.fragment_delta}")
    if args.plan:
        written = write_plan(report, Path(args.plan))
        logging.success(f"Wrote {report.edits}-edit recovery plan: {written}")
        if report.warnings:
            logging.warning(
                "The plan is INCOMPLETE - re-run the scripts for the operations listed above")


def cmd_motiongraph_chooser(args: argparse.Namespace) -> None:
    """List, grow, or re-weight a RandomAnimationActivity chooser.

    A chooser picks among clips BY NAME, so adding one needs no Activity wrapper,
    no payload, no state and no edge - just a name string and one more array slot.
    That is what makes "give this species another idle/preen/rest" a small edit.

    Weights are relative; the engine draws from their sum. The share is printed so
    the number you want ("about 15%") maps to a weight without arithmetic.
    """
    from pathlib import Path
    from source.formats.motiongraph.chooser_growth import (
        grow_random_animation_chooser, list_choosers, set_chooser_weights)

    game = resolve_game_label(args.game)
    source = Path(args.ovl).resolve()
    ensure_exists(str(source), "file")

    if not args.add and not args.weights:
        try:
            rows = list_choosers(source, args.name, game)
        except Exception as exc:
            die(f"could not read choosers: {exc}")
        if args.match:
            needle = args.match.casefold()
            rows = [r for r in rows
                    if any(needle in c["short"].casefold() for c in r["clips"])]
        for row in rows:
            print(f"chooser {row['pool']}:{row['offset']}  {row['count']} clips  "
                  f"blend {row['blend_time']:.2f}  flags {row['flags']}"
                  + ("   (flags 8 ignores blend_time)" if row["flags"] == 8 else ""))
            for index, clip in enumerate(row["clips"]):
                print(f"    [{index}] {clip['short']:28s} weight {clip['weight']:4d}"
                      f"  {clip['share']:6.1%}")
        if not rows:
            print("no choosers matched")
        return

    if not args.output:
        die("--add and --weights write a file; pass -o/--output (a staged family)")
    output = Path(args.output).resolve()
    if args.chooser is None:
        die("--add and --weights need --chooser POOL:OFFSET (see the plain listing)")
    try:
        pool_text, offset_text = str(args.chooser).split(":")
        chooser_pool, chooser_offset = int(pool_text), int(offset_text)
    except ValueError:
        die(f"--chooser must look like POOL:OFFSET, got {args.chooser!r}")

    if args.add:
        try:
            report = grow_random_animation_chooser(
                source, output, chooser_pool, chooser_offset, args.add,
                weight=args.weight, name=args.name, game=game)
        except Exception as exc:
            die(f"could not add the clip: {exc}")
        logging.success(f"{report.entries_before} -> {report.entries_after} clips; "
                        f"array {report.old_array} -> {report.new_array}; "
                        f"+{report.pool_growth} bytes")
        for clip in report.names:
            print("   ", clip)
        logging.warning("Cobra reload is not proof - verify with motiongraph-census "
                        "--against and then in game")
        source = output

    if args.weights:
        try:
            weights = [int(x) for x in str(args.weights).replace(" ", "").split(",") if x]
        except ValueError:
            die(f"--weights must be a comma-separated integer list, got {args.weights!r}")
        if source != output and not output.is_file():
            die("stage the complete OVL family at --output first")
        try:
            row = set_chooser_weights(source if source != output else args.ovl,
                                      output, chooser_pool, chooser_offset,
                                      weights, args.name, game)
        except Exception as exc:
            die(f"could not set weights: {exc}")
        logging.success("weights updated")
        for index, clip in enumerate(row["clips"]):
            print(f"    [{index}] {clip['short']:28s} weight {clip['weight']:4d}"
                  f"  {clip['share']:6.1%}")


def cmd_motiongraph_census(args: argparse.Namespace) -> None:
    """Census decoded motiongraph objects, optionally diffing two builds.

    A decoded object is only reachable through a live pointer chain, so an edit
    that redirects an allocation's last inbound reference silently removes it
    from the graph while every size and reload check still passes. Diffing two
    censuses is the check that catches it.
    """
    from pathlib import Path
    from source.formats.motiongraph.capacity import census, diff_census
    from source.formats.motiongraph.edit import load_motiongraph

    game = resolve_game_label(args.game)
    source = Path(args.ovl).resolve()
    ensure_exists(str(source), "file")
    try:
        _, loader = load_motiongraph(source, args.name, game)
        current = census(loader)
    except Exception as exc:
        die(f"motiongraph census failed: {exc}")
    if not args.against:
        print(f"total_decoded={current['total']}")
        for name, count in sorted(current["counts"].items(), key=lambda row: (-row[1], row[0])):
            print(f"{count:8d}  {name}")
        return

    baseline_path = Path(args.against).resolve()
    ensure_exists(str(baseline_path), "file")
    try:
        _, baseline_loader = load_motiongraph(baseline_path, args.name, game)
        difference = diff_census(census(baseline_loader), current)
    except Exception as exc:
        die(f"motiongraph census diff failed: {exc}")
    print(
        f"total {difference['total_before']} -> {difference['total_after']} "
        f"(added={difference['added']} removed={difference['removed']})"
    )
    for name, row in sorted(difference["changed_types"].items()):
        print(f"{name}: {row['before']} -> {row['after']}")
        for address in row["added"]:
            print(f"    + {address[0]}:{address[1]}")
        for address in row["removed"]:
            print(f"    - {address[0]}:{address[1]}")
    if not difference["changed_types"]:
        print("no decoded object changed")
    if difference["removed"]:
        logging.warning(
            "%d decoded object(s) disappeared - their allocations are now orphaned "
            "and their bytes are absorbed into the preceding allocation",
            difference["removed"],
        )


def cmd_motiongraph_capacity(args: argparse.Namespace) -> None:
    """Audit allocation slack and graph objects without modifying the source."""
    from pathlib import Path
    from source.formats.motiongraph.capacity import build_capacity_audit, render_capacity_markdown
    from source.formats.motiongraph.edit import load_motiongraph

    source = Path(args.ovl).resolve()
    output = Path(args.output).resolve()
    ensure_exists(str(source), "file")
    try:
        _, loader = load_motiongraph(source, args.name, resolve_game_label(args.game))
        audit = build_capacity_audit(loader, source)
        report = render_capacity_markdown(audit)
    except Exception as exc:
        die(f"motiongraph capacity audit failed: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logging.success("Wrote topology-capacity report: %s", output)
    if args.json:
        json_path = Path(args.json).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        logging.success("Wrote topology-capacity JSON: %s", json_path)
    summary, reach = audit["summary"], audit["reachability"]
    print(
        f"allocations={summary['allocations']} "
        f"reusable_arrays={summary['counted_arrays_with_reusable_slots']} "
        f"candidate_slots={summary['reusable_array_slots']} "
        f"adjacent_growth={summary['adjacent_growth_candidates']} "
        f"dormant_activities={len(reach['dormant_activities'])} "
        f"no_inbound_states={len(reach['no_known_inbound_states'])}"
    )


def cmd_motiongraph_grow_null_prefix(args: argparse.Namespace) -> None:
    """Run the fixed-pool null-prefix topology-growth experiment."""
    from pathlib import Path
    from source.formats.motiongraph.surgical_growth import grow_null_prefix

    _check_staged_output(args)
    try:
        report = grow_null_prefix(
            Path(args.ovl), Path(args.output), args.array_pool, args.array_offset,
            name=args.name, game=resolve_game_label(args.game),
            allow_eight_alignment=args.allow_8_byte_alignment,
        )
    except Exception as exc:
        die(f"motiongraph null-prefix growth failed: {exc}")
    logging.success(
        "Grew counted array %d:%d -> %d from %d to %d entries",
        report.pool, report.old_offset, report.new_offset,
        report.old_count, report.new_count,
    )
    logging.success(
        "Fixed archive topology retained: %d pools, %d fragments, %d uncompressed bytes",
        report.pools, report.fragments, report.uncompressed_size,
    )
    logging.success("Wrote and decoded staged experiment: %s", report.output)


def _check_staged_output(args: argparse.Namespace):
    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.force:
        die(f"Output already exists: {output} (use --force to replace the staged copy)")


def cmd_motiongraph_retarget_clip(args: argparse.Namespace) -> None:
    from pathlib import Path
    from source.formats.motiongraph.static_patch import repoint_existing_string

    _check_staged_output(args)
    try:
        report = repoint_existing_string(
            Path(args.ovl), Path(args.output), args.source_string, args.target_string,
            expected_count=args.expect_count, game=resolve_game_label(args.game),
        )
    except Exception as exc:
        die(f"motiongraph clip retarget failed: {exc}")
    logging.success(
        "Repointed %d fragments: pool %d, %d -> %d (%d changed bytes)",
        report.references, report.source_pool, report.source_offset,
        report.target_offset, report.changed_bytes,
    )
    logging.success("Topology unchanged: %s", report.topology)
    logging.success("Wrote and reloaded staged family: %s", report.output)


def cmd_motiongraph_string_slot(args: argparse.Namespace) -> None:
    from pathlib import Path
    from source.formats.motiongraph.static_patch import patch_string_slot

    _check_staged_output(args)
    try:
        report = patch_string_slot(
            Path(args.ovl), Path(args.output), args.source_string, args.target_string,
            expected_offset=args.expect_offset, game=resolve_game_label(args.game),
        )
    except Exception as exc:
        die(f"motiongraph string-slot patch failed: {exc}")
    logging.success(
        "Patched string slot in pool %d at %d (%d changed bytes)",
        report.source_pool, report.source_offset, report.changed_bytes,
    )
    logging.success("Topology unchanged: %s", report.topology)
    logging.success("Wrote and reloaded staged family: %s", report.output)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command-line OVL tool using cobra-tools' OvlFile."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    game_vals = game_choices()
    comp_vals = compression_choices()

    # new
    p_new = sub.add_parser(
        "new",
        help="Create a new OVL from a folder (File > New from folder).",
    )
    p_new.add_argument(
        "-g", "--game",
        help="Game identifier (matches GUI 'Game' dropdown).",
        choices=game_vals if game_vals else None,
        required=True,
    )
    p_new.add_argument(
        "-i", "--input",
        help="Input folder containing files to pack into the OVL.",
        required=True,
    )
    p_new.add_argument(
        "-o", "--output",
        help="Output .ovl file path.",
        required=True,
    )
    p_new.add_argument(
        "-c", "--compression",
        help="Compression method (matches GUI 'Compression').",
        choices=comp_vals if comp_vals else None,
    )
    p_new.add_argument(
        "--update-aux",
        action="store_true",
        help="Set commands['update_aux']=True when saving.",
    )
    p_new.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    p_new.set_defaults(func=cmd_new)

    # extract
    p_ext = sub.add_parser(
        "extract",
        help="Extract files from an OVL.",
    )
    p_ext.add_argument(
        "ovl",
        help="Path to the .ovl file to extract.",
    )
    p_ext.add_argument(
        "-o", "--output",
        help=(
            "Output folder (created if missing). "
            "If omitted, a folder named after the OVL file will be created "
            "next to the OVL (e.g. Main.ovl -> Main/)."
        )
    ),
    p_ext.add_argument(
        "-g", "--game",
        help="Game identifier (optional; if omitted, OvlFile may auto-detect).",
        choices=game_vals if game_vals else None,
    )
    p_ext.add_argument(
        "--type",
        action="append",
        default=[],
        help="Restrict extraction to specific file types/extensions (can repeat).",
    )
    p_ext.add_argument(
        "--name",
        action="append",
        default=[],
        help="Restrict extraction to specific internal entry names (can repeat).",
    )
    p_ext.set_defaults(func=cmd_extract)

    # inject
    p_inj = sub.add_parser(
        "inject",
        help="Inject/replace files into an OVL.",
    )
    p_inj.add_argument(
        "ovl",
        help="Path to the .ovl file to modify.",
    )
    p_inj.add_argument(
        "-g", "--game",
        help="Game identifier (matches GUI 'Game' dropdown).",
        choices=game_vals if game_vals else None,
        required=True,
    )
    p_inj.add_argument(
        "-i", "--input",
        help="Folder whose contents will be injected (recursively).",
    )
    p_inj.add_argument(
        "-f", "--file",
        action="append",
        default=[],
        help="Individual file path(s) to inject. Can be repeated.",
    )
    p_inj.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the OVL in place (overwrite the input file).",
    )
    p_inj.add_argument(
        "-o", "--output",
        help="Output .ovl file path (required unless --in-place).",
    )
    p_inj.add_argument(
        "--update-aux",
        action="store_true",
        help="Set commands['update_aux']=True when saving.",
    )
    p_inj.add_argument(
        "--update",
        action="store_true",
        help="Write into the existing loaders instead of removing and re-creating them. "
             "Keeps the OVL's pool and data entry layout, which JWE3 requires. The file "
             "must already exist in the OVL and keep the same shape.",
    )
    p_inj.set_defaults(func=cmd_inject)

    # retarget-family
    p_ret = sub.add_parser(
        "retarget-family",
        help="DEPRECATED - fixed-width djb2 alias only. Use the GUI's Rename Species Family (source/formats/ovl/species_rename.py); different-length renames are game-verified and this width constraint is retired.",
    )
    p_ret.add_argument(
        "ovl",
        help="Donor .ovl; all sibling .ovs companions and referenced AUX files are staged.",
    )
    p_ret.add_argument(
        "-o", "--output",
        required=True,
        help="Output .ovl path. Companion files are written beside it.",
    )
    p_ret.add_argument(
        "-g", "--game",
        choices=game_vals if game_vals else None,
        default="Jurassic World Evolution 3",
        help="Game identifier; currently restricted to Jurassic World Evolution 3.",
    )
    p_ret.add_argument(
        "--from", dest="donor", required=True,
        help="Lowercase donor prefix, for example deinosuchus.",
    )
    p_ret.add_argument(
        "--to", dest="alias", required=True,
        help="Equal-width lowercase alias with the same djb2 hash state.",
    )
    p_ret.add_argument(
        "--force", action="store_true",
        help="Replace an existing output family, but never the donor family.",
    )
    p_ret.set_defaults(func=cmd_retarget_family)

    # motiongraph-fields
    p_mgf = sub.add_parser(
        "motiongraph-fields",
        help="Inspect verified fixed-width motiongraph activity fields and make patch plans.",
    )
    p_mgf.add_argument("ovl", help="Source OVL containing the motiongraph.")
    p_mgf.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgf.add_argument(
        "--name", help="Internal .motiongraph name; auto-detected when exactly one exists.",
    )
    p_mgf.add_argument("--activity", help="Case-insensitive substring of the activity label.")
    p_mgf.add_argument("--activity-type", help="Exact activity payload type.")
    p_mgf.add_argument("--field", help="Exact payload field path, for example speed.float.")
    p_mgf.add_argument("--json", help="Optional JSON field-report output path.")
    p_mgf.add_argument("--set", dest="set_value", type=float, help="Set a scalar or curve value.")
    p_mgf.add_argument(
        "--set-flag", action="append", default=[], metavar="NAME=0|1",
        help="Set or clear one named bitfield flag; repeatable.",
    )
    p_mgf.add_argument("--set-enum", help="Select a named enum value.")
    p_mgf.add_argument("--plan", help="Write a self-verifying JSON patch plan.")
    p_mgf.set_defaults(func=cmd_motiongraph_fields)

    # motiongraph-apply
    p_mga = sub.add_parser(
        "motiongraph-apply",
        help="Apply a fixed-size plan to STATIC without rebuilding motiongraph topology.",
    )
    p_mga.add_argument("ovl", help="Source OVL; it is never overwritten.")
    p_mga.add_argument("-o", "--output", required=True, help="Staged output OVL.")
    p_mga.add_argument("--plan", required=True, help="Plan from motiongraph-fields.")
    p_mga.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mga.add_argument("--force", action="store_true", help="Replace an existing staged output.")
    p_mga.set_defaults(func=cmd_motiongraph_apply)

    # motiongraph-report
    p_mgr = sub.add_parser(
        "motiongraph-report",
        help="Render a human-readable state/activity graph or MRF decision tree.",
    )
    p_mgr.add_argument("ovl", help="Source OVL containing the motiongraph.")
    p_mgr.add_argument("-o", "--output", required=True, help="Markdown output path.")
    p_mgr.add_argument("--kind", choices=("state", "decision"), default="state")
    p_mgr.add_argument(
        "--name", help="Internal .motiongraph name; auto-detected when exactly one exists.",
    )
    p_mgr.add_argument("--json", help="Optional structured output for --kind state.")
    p_mgr.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgr.set_defaults(func=cmd_motiongraph_report)

    # motiongraph-capacity
    p_mgc = sub.add_parser(
        "motiongraph-capacity",
        help="Audit reusable allocation slack and dormant motiongraph topology.",
    )
    p_mgc.add_argument("ovl", help="Source OVL containing the motiongraph; never modified.")
    p_mgc.add_argument("-o", "--output", required=True, help="Markdown report path.")
    p_mgc.add_argument("--json", help="Optional complete structured audit path.")
    p_mgc.add_argument(
        "--name", help="Internal .motiongraph name; auto-detected when exactly one exists.",
    )
    p_mgc.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgc.set_defaults(func=cmd_motiongraph_capacity)

    # motiongraph-census
    p_mgn = sub.add_parser(
        "motiongraph-census",
        help="Count decoded motiongraph objects; --against diffs two builds.",
    )
    p_mgn.add_argument("ovl", help="OVL containing the motiongraph; never modified.")
    p_mgn.add_argument(
        "--against", help="Baseline OVL to diff against, e.g. the build this one was made from.",
    )
    p_mgn.add_argument(
        "--name", help="Internal .motiongraph name; auto-detected when exactly one exists.",
    )
    p_mgn.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgn.set_defaults(func=cmd_motiongraph_census)

    # motiongraph-chooser
    p_mgh = sub.add_parser(
        "motiongraph-chooser",
        help="List / grow / re-weight a random animation chooser (rest, preen, eat...).",
    )
    p_mgh.add_argument("ovl", help="Source OVL; never modified unless -o is given.")
    p_mgh.add_argument("--match", help="Only list choosers containing this clip substring.")
    p_mgh.add_argument("--chooser", help="Target chooser as POOL:OFFSET, from the listing.")
    p_mgh.add_argument("--add", help="Clip name to append, e.g. 'Species$Rest03'.")
    p_mgh.add_argument("--weight", type=int, help="Weight for the added clip.")
    p_mgh.add_argument("--weights", help="Comma-separated weights for every clip, in order.")
    p_mgh.add_argument("-o", "--output", help="Staged same-named OVL to write.")
    p_mgh.add_argument("--name", help="Internal .motiongraph name; auto-detected.")
    p_mgh.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgh.set_defaults(func=cmd_motiongraph_chooser)

    # motiongraph-diff
    p_mgd = sub.add_parser(
        "motiongraph-diff",
        help="Diff two builds into a replayable patch plan (motiongraph recovery).",
    )
    p_mgd.add_argument("ovl", help="Baseline OVL to replay ONTO; never modified.")
    p_mgd.add_argument(
        "--against", required=True, help="Edited OVL whose changes should be captured.",
    )
    p_mgd.add_argument("--plan", help="Write the patch plan here.")
    p_mgd.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgd.set_defaults(func=cmd_motiongraph_diff)

    # motiongraph-grow-null-prefix
    p_mgg = sub.add_parser(
        "motiongraph-grow-null-prefix",
        help="Experimental: prepend a null array entry using adjacent padding.",
    )
    p_mgg.add_argument("ovl", help="Pristine source OVL; never overwritten.")
    p_mgg.add_argument("-o", "--output", required=True, help="Same-named OVL in a full stage.")
    p_mgg.add_argument("--array-pool", type=int, required=True, help="Global pool index.")
    p_mgg.add_argument("--array-offset", type=int, required=True, help="Current array offset.")
    p_mgg.add_argument(
        "--allow-8-byte-alignment", action="store_true",
        help="Explicitly permit moving an 8-byte element array to an 8-mod-16 target.",
    )
    p_mgg.add_argument(
        "--name", help="Internal .motiongraph name; auto-detected when exactly one exists.",
    )
    p_mgg.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgg.add_argument("--force", action="store_true", help="Replace the staged main OVL.")
    p_mgg.set_defaults(func=cmd_motiongraph_grow_null_prefix)

    # motiongraph-retarget-clip
    p_mgret = sub.add_parser(
        "motiongraph-retarget-clip",
        help="Repoint matching fragments to another existing same-pool clip string.",
    )
    p_mgret.add_argument("ovl", help="Pristine source OVL; never overwritten.")
    p_mgret.add_argument("-o", "--output", required=True, help="Same-named OVL in a full staged family.")
    p_mgret.add_argument("--from", dest="source_string", required=True, help="Exact existing source string.")
    p_mgret.add_argument("--to", dest="target_string", required=True, help="Exact existing target string.")
    p_mgret.add_argument("--expect-count", type=int, required=True, help="Required source fragment count.")
    p_mgret.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgret.add_argument("--force", action="store_true", help="Replace the staged OVL.")
    p_mgret.set_defaults(func=cmd_motiongraph_retarget_clip)

    # motiongraph-string-slot
    p_mgstr = sub.add_parser(
        "motiongraph-string-slot",
        help="Replace one unique STATIC string without growing its allocation.",
    )
    p_mgstr.add_argument("ovl", help="Pristine source OVL; never overwritten.")
    p_mgstr.add_argument("-o", "--output", required=True, help="Same-named OVL in a full staged family.")
    p_mgstr.add_argument("--from", dest="source_string", required=True, help="Exact unique source string.")
    p_mgstr.add_argument("--to", dest="target_string", required=True, help="ASCII replacement fitting the source slot.")
    p_mgstr.add_argument("--expect-offset", type=int, help="Optional required source offset.")
    p_mgstr.add_argument(
        "-g", "--game", default="Jurassic World Evolution 3",
        choices=game_vals if game_vals else None,
    )
    p_mgstr.add_argument("--force", action="store_true", help="Replace the staged OVL.")
    p_mgstr.set_defaults(func=cmd_motiongraph_string_slot)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    finally:
        # The CLI logger is asynchronous. Drain it so short commands do not lose
        # their final success/error summary when the Python process exits.
        listener = get_global_listener()
        if listener is not None:
            listener.stop()


if __name__ == "__main__":
    main()
