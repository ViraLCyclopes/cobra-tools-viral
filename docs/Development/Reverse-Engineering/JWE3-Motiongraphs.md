# JWE3 motiongraph editing

JWE3 motiongraphs can be read, reported, and edited safely when an edit keeps
the existing STATIC allocation and fragment topology. They are animation
decision graphs: MANI files contain bone motion; motiongraphs select clips,
blend activities, and emit timed signals.

## Proven boundary

| Operation | Confidence |
|---|---|
| Decode state, activity, decision, locomotion, and data-stream structures | Cross-game corpus verified |
| Change fixed-width scalar values such as animation playback speed | Game verified |
| Repoint fragment references to another existing same-pool clip string | Game verified |
| Replace text inside an existing string allocation | Game verified |
| Retarget a complete family with an equal-width DJB2 collision alias | Game verified |
| Patch bitfields, enums, and curve values at verified addresses | Byte verified; needs targeted game tests |
| Add/remove nodes, grow pools, or rebuild from extracted XML | Rejected by the game |

Pool growth with no semantic change raises the game's `0x000FDEAD` assertion.
Do not use an XML round trip as an injection workflow. The supported commands
patch the compressed STATIC archive directly and preserve its layout.

## Safety workflow

Always keep a pristine source family and make a complete staging copy containing
the `.ovl`, every `.ovs*` sibling, and its `.aux` file. Give the staged OVL the
same basename as the source. The patch commands refuse to overwrite the source,
then reload the staged family after writing it.

In PowerShell, animation names containing `$` must use **single quotes**. A
double-quoted `Acrocanthosaurus$StandIdle01` is expanded by PowerShell and may
silently become only `Acrocanthosaurus`.

## GUI editor

Launch `motiongraph_tool_gui.py` directly, or choose **Utility > Motiongraph
Editor** in the OVL Tool. The editor provides:

- a directed state graph with selectable nodes, transition arrows, search,
  unrestricted pan/zoom, edge hiding, fit-to-view, and `H` to recenter;
- a decision graph showing opcodes, variables, result branches, and state outputs;
- a searchable hierarchical activity tree for each selected state;
- exact STATIC pool/offset drill-down from a selected activity-tree node into
  only that instance's byte-verified editable fields;
- explicit field-scan scope for the entire motiongraph, one exact activity, or
  the activities reachable from one or more Ctrl/Shift-selected states;
- one/two-hop parent-and-child focus, double-click focus navigation, Show All
  breadcrumbs, and jumpable shared-activity references;
- searchable state/activity and decision-tree views;
- verified fixed-width field discovery and patch-plan creation;
- existing-string clip retargeting and fixed-allocation string replacement;
- a full-family staging copy and post-write reload verification.

The GUI deliberately has no generic XML import and never writes into the live
game. Each Apply currently regenerates the staged main OVL from the pristine
source, so it replaces any earlier edit in that same stage directory. Use a
separate stage for each experiment, or alternate complete source/stage copies
when chaining operations. State names prefixed with `≈` are inferred labels,
not authored Frontier identifiers.

## Putting a custom motiongraph into a species mod

A custom motiongraph is currently a modified donor graph, not a new graph built
from an empty file. Choose a shipped donor with suitable states and activities,
copy its entire archive family, and repurpose existing values, clip references,
and string allocations. Every referenced clip must exist in the target MANI
set. The species prefab and asset package must load that staged archive family
and select its motiongraph.

When the family itself must be renamed for a new species, use the verified
`retarget-family` collision-alias workflow and update the prefab/package
references together. Arbitrary ordinary renames are not safe yet: JWE3's AUX
and internal hash relationships make a superficially successful rename capable
of crashing during load. Equal-width lowercase DJB2 collision aliases preserve
the required identity while providing a distinct package name.

This supports genuinely custom routing and clip selection within a donor's
existing slots. It does **not** yet support adding a state, activity, decision
node, transition, string allocation, or a larger graph. Those operations grow
or rebuild STATIC topology and are rejected by the engine.

## Inspect and edit values

List all verified speed fields for a clip:

```powershell
python ovl_tool_cmd.py motiongraph-fields STOCK.ovl `
  --activity 'Acrocanthosaurus$StandIdle01' `
  --activity-type AnimationActivity `
  --field speed.float
```

Create a self-verifying patch plan without changing the OVL:

```powershell
python ovl_tool_cmd.py motiongraph-fields STOCK.ovl `
  --activity 'Acrocanthosaurus$StandIdle01' `
  --activity-type AnimationActivity `
  --field speed.float --set 1.25 `
  --plan C:\Temp\acro_idle_speed.json
```

Apply it to the same-named OVL in a complete staged family:

```powershell
python ovl_tool_cmd.py motiongraph-apply STOCK.ovl `
  --output C:\Temp\AcroStage\Acrocanthosaurus_Female.ovl `
  --plan C:\Temp\acro_idle_speed.json --force
```

Every plan records the old bytes at each address. Application stops if the
source no longer matches, a field width changes, edits overlap, a target is not
in STATIC, or any byte outside the intended set changes.

Supported value kinds are:

- fixed-width numeric scalars;
- named enum members;
- individual named bitfield flags;
- biased-bfloat16 curve values (quantized like Frontier's files).

Use `--set-enum NAME` or repeat `--set-flag NAME=0|1` instead of `--set` for
enums and bitfields.

## Retarget clips without adding strings

Repoint all fragment references from one existing string to another existing
string in the same STATIC pool:

```powershell
python ovl_tool_cmd.py motiongraph-retarget-clip STOCK.ovl `
  --output C:\Temp\AcroStage\Acrocanthosaurus_Female.ovl `
  --from 'Acrocanthosaurus$StandIdle01' `
  --to 'Acrocanthosaurus$StandPreen' `
  --expect-count 160 --force
```

`--expect-count` prevents a version mismatch or overly broad selection from
silently changing the wrong number of references. Cross-pool retargeting is
refused because it has not been game verified.

To rename a unique existing string within its current allocation:

```powershell
python ovl_tool_cmd.py motiongraph-string-slot STOCK.ovl `
  --output C:\Temp\AcroStage\Acrocanthosaurus_Female.ovl `
  --from 'Acrocanthosaurus$StandIdle01' `
  --to 'Acrocanthosaurus$StandIdle02' `
  --expect-offset 4760 --force
```

The replacement must be ASCII and no longer than the original including its
null terminator. Shorter replacements are null padded. This only changes the
motiongraph reference; the target MANI clip must exist separately.

## Reports

Generate the anonymous state/activity graph:

```powershell
python ovl_tool_cmd.py motiongraph-report STOCK.ovl `
  --kind state --output C:\Temp\states.md `
  --json C:\Temp\states.json
```

Generate the complete MRF decision tree, including transition-embedded nodes:

```powershell
python ovl_tool_cmd.py motiongraph-report STOCK.ovl `
  --kind decision --output C:\Temp\decisions.md
```

State names in these reports are derived from their shallowest distinctive clip
or data-stream signal. They are not authored IDs; shipped JWE3 states are
anonymous and positional.

## Known semantic unknowns

The following fields decode consistently but do not yet have proven gameplay
meanings: `TransitionConditionRecord.activity_flags`, transition fields at
offsets 36/44/56/60, `MRFMember1.count_0` and `count_4`, the singleton `Error`
and `StateTransitionWait` opcode behavior, and two neutral bytes in
`ForwardActivitySmoothTransitionData`. None blocks extraction, reporting,
fixed-width value edits, or existing-string retargeting. Keep their labels
neutral until a controlled game experiment identifies them.

## Motiongraph limits versus animation scale

Motiongraph completion does not add ACL scale encoding. Existing compressed MANI
scale tracks can be preserved or patched without changing their topology, and
whole-creature prefab `UniversalScale` is separate. General Blender-authored
bone scale still requires reverse-engineering Frontier's customized ACL encoder.

## Locomotion speed and audio

A controlled ViralSarco test showed that `SpeciesAnimation.WalkSpeed=9` and
`RunSpeed=25` were present in the FDB extracted from the installed `Main.ovl`,
yet a female ViralSarco and female Deinosuchus still travelled at the same
speed. Both use the Deinosuchus locomotion/MANI basis. Those FDB columns are
therefore expected-speed metadata for gameplay/planning, not the physical
displacement source.

`Locomotion2BlendSpaceNode.speed` is not metres per second either: shipped slow
Patagotitan and fast Gallimimus graphs both use normalized `0.0` on-spot and
`1.0` moving samples driven by a `Speed` variable. Primary Walk/Run clips are
selected by `Locomotion2Activity`, not ordinary `AnimationActivity` nodes; in
the ViralSarco graph, a `Walk` field scan only found the transitional
`SurfaceSwimToWalk` playback rate. The leading physical-speed candidate is thus
motion-extracted translation stored with the MANI clips (or a native locomotion
controller input not serialized in the graph), rather than an editable primary
Walk/Run playback field.

Motiongraphs do not contain audio samples or Wwise banks. Their
`DataStreamProducerActivity` nodes emit named curves with stream name, type,
bone, and location fields; `AudioBlend` streams can drive an audio RTPC or
blend. Custom sound still requires a loaded BNK/event and a compatible consumer.
Within the fixed-topology boundary, an existing data-stream or triggered-activity
slot may be retargeted, but adding an entirely new audio node is not yet safe.
