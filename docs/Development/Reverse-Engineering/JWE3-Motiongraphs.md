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
| Add one fragment/reference without changing pools | Game verified |
| Grow a pool without updating `ArchiveMeta.unk_0` | Game rejected; original Test A was incomplete |
| Grow pool 0 with its runtime reservation | Game rejected (`0x000FDEAD`, Test A2) |
| Grow the last STATIC pool with its runtime reservation | Game verified (Test A3) |
| Grow structural pool 105 and relocate 28 later pools | Game verified (Test A4) |
| Append an Activity past a full 16,416-byte type-2 pool page | Game rejected (`+0x167EC0A`, Tests F/F2) |
| Append an Activity in the partial final type-2 pool page | Game verified (Test F3) |
| Rebuild from extracted XML | Rejected by the game |

The original inert pool-growth test raised the game's `0x000FDEAD` assertion and
omitted a matching increase to `ArchiveMeta.unk_0`. Test A2 corrected that omission
but raised the same assertion when pool 0 grew. Test A3 instead grew the last pool
with the same metadata corrections and loaded successfully. Therefore buffer-block
relocation and physical pool growth are not universally rejected; pool 0 or later-
pool relocation caused A2. Test A4 then grew structural pool 105 and relocated 28
later pools successfully, isolating the rejection to pool 0's special MotionGraph
string-pool semantics. Do not use an XML round trip as an injection workflow.

Test F appended a 48-byte `Activity` wrapper to pool 67 and added its two
internal fragments, but crashed at `JWE3.exe+0x167EC0A`. Test F2 changed
`MemPool.num_files` from 342 to 343 and crashed at the exact same instruction;
that field is not proven to be an Activity allocation count.

The stronger invariant is the type-2 pool page boundary. Pools 62, 63, 65, 66,
67, and 68 are each exactly 16,416 bytes, while pool 69 is the final partial
type-2 page at 2,568 bytes. F/F2 extended the already-full pool 67 to 16,464.
Test F3 instead aligns and appends the wrapper in partial pool 69 without
changing `num_files`. Acro loaded, spawned, and ran smoothly. This is the first
game-verified newly allocated Activity wrapper and establishes partial-tail-page
placement as the current safe allocation rule. Creating a new page remains
unproven.

Test G is the next gate: its F3 wrapper points to a newly cloned 32-byte
`ToggledActivityActivityData` in the final partial type-3 pool. The first offline
build exposed another required invariant: 1,463 empty arrays shared the old pool
end as a sentinel. Their fragment targets must move to the new pool end before
the old end becomes object storage. The corrected build semantically reloads as
the intended data type, and Acro loaded and spawned successfully. New Activity
wrappers plus independently allocated ActivityData are therefore game verified.

### Fixed-pool topology-growth research

Both the original physical-pool append and Test A2 with a correspondingly larger
runtime reservation were game-rejected. Separately, `motiongraph-capacity` audits
decoded pointer targets,
known runtime reachability, and verified zero padding without changing the source.
A pristine land/air/marine corpus found:

- zero counted arrays with a whole unused element already inside their allocation;
- zero activities outside known state and transition-local runtime roots;
- two land, one air, and two marine arrays that could consume eight bytes of
  immediately preceding padding while keeping all pool sizes fixed;
- none of those candidates preserves cobra's usual 16-byte allocation alignment;
  each new target is still naturally 8-byte aligned for an `ActivityReference`.

`motiongraph-grow-null-prefix` implements the smallest falsifiable experiment. It
moves one counted-array boundary back by one 8-byte element, increments its existing
count, and prepends a null reference. It does not add bytes, add a fragment, move any
existing element, or change pool/archive/fragment counts. The pristine Acro candidate
changes exactly two bytes in uncompressed STATIC and reloads with State 151 decoding
as 16 entries.

**Game-rejected 2026-08-26:** JWE3 reached the menu, loaded a map, spawned Acro,
and initially ran for more than one minute, but then crashed without any further
install change. The delayed failure was an access violation at
`JWE3.exe+0x1B0A117`, reading `0x10` from a null base. Logical count growth with a
prepended null entry at this 8-byte-aligned target must therefore be treated as
unsafe. Test D's non-null-reference experiment was never installed.

The revised Test D removes the null-reference confounder without growing the
fragment table: it moves one existing activity fragment from a state with no known
inbound edge into State 151's new slot and changes the donor count from two to one.
Offline reload verifies that both arrays contain only non-null references, the
total logical reference count is unchanged, and all archive topology measurements
remain fixed.

**Game-verified 2026-08-26:** the revised Test D loaded, spawned Acro, and remained
stable for approximately five to ten minutes at maximum game speed. This verifies
the tested 8-byte-aligned boundary/count change when the added slot contains a real
reference, and strongly identifies Test C's null reference as its delayed crash
cause. It proves fixed-pool topology *redistribution*, not net growth: the total
logical reference count and fragment count remained unchanged. A genuinely new
fragment record is the next unproven gate.

Test E implements that gate without changing any pool. Starting from Test C's
16-entry State 151, it appends one fragment from the new slot to an existing shared
`ToggledActivityActivity`. The fragment count becomes 96,743 and uncompressed
STATIC grows by exactly the 16-byte fragment-record size; all 129 pools remain
byte-identical. The staged archive reloads with 16 non-null State 151 entries and
passes validation/collection.

**Game-verified 2026-08-26:** Test E loaded and remained smooth through the full
observation window at maximum game speed plus additional simulation acceleration
from Kai's speed controls. No crash or animation corruption occurred. JWE3 therefore
accepts this genuine one-record fragment-table increase and one net counted activity
reference. This still does not add storage for a new activity object; fixed-pool
object placement is the next distinct topology-growth gate.

```powershell
python ovl_tool_cmd.py motiongraph-capacity STOCK.ovl `
  -o C:\Temp\capacity.md --json C:\Temp\capacity.json

python ovl_tool_cmd.py motiongraph-grow-null-prefix STOCK.ovl `
  -o C:\Temp\Stage\Acrocanthosaurus_Female.ovl `
  --array-pool 105 --array-offset 12016 --allow-8-byte-alignment --force
```

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
  unrestricted pan/zoom, edge hiding, fit-to-view, and `H` to recenter. Focused
  neighborhoods can use a left-to-right hierarchical layout (incoming states,
  selected state, outgoing states), expose one through four hops, or retain the
  original compact overview grid;
- a decision graph showing opcodes, variables, result branches, and state outputs;
- a searchable hierarchical activity tree for each selected state;
- exact STATIC pool/offset drill-down from a selected activity-tree node into
  only that instance's byte-verified editable fields;
- explicit field-scan scope for the entire motiongraph, one exact activity, or
  the activities reachable from one or more Ctrl/Shift-selected states;
- one/two-hop parent-and-child focus, double-click focus navigation, Show All
  breadcrumbs, and jumpable shared-activity references;
- searchable state/activity and decision-tree views;
- verified fixed-width field discovery and a cumulative mixed-value patch queue;
- existing-string clip retargeting and fixed-allocation string replacement;
- a full-family staging copy and post-write reload verification.

The GUI deliberately has no generic XML import and never writes into the live
game. Add every desired field/value operation to the queue, optionally save the
queue as JSON, and then Apply once. The editor rejects conflicting or overlapping
addresses and regenerates the staged main OVL atomically from the pristine source.
A later Apply still starts from that source, so update the queue rather than
expecting it to build on an earlier staged output. State names prefixed with `≈`
are inferred labels, not authored Frontier identifiers.

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

Direct ACL decoding now identifies the clip-side quantity precisely. Shipped
locomotion clips carry `X/Y/Z Motion Track` scalar channels alongside matching
`srb` and `def_c_root_joint` translation. Forward distance per cycle is the
endpoint delta of `Z Motion Track`. Examples (distance / duration = authored
track rate): Gallimimus Walk `3.791670 / 1.166666 = 3.250005`, Patagotitan Walk
`4.430119 / 6.0 = 0.738353`, and Acro Walk `7.791353 / 1.8 = 4.328530`.

The FDB values are not the distance-per-cycle storage. They often approximate
the decoded rate, but do not universally equal it: Patagotitan Run decodes to
`6.848213 / 3.333333 = 2.054464` while its FDB `RunSpeed` is `3.49`; Acro Run
decodes to `12.763560 / 0.933333 = 13.675248` while its FDB value is `11.86`.
This implies a second runtime/controller reconciliation step or planning
metadata. The ViralSarco race, where large installed FDB changes did not change
world speed relative to the inherited Deinosuchus basis, means the FDB row must
not yet be described as the final runtime authority.

Test N is the clean discriminator: a size-preserving compressed patch of
Patagotitan Walk `srb` translation range only at 10x Z, leaving `Z Motion Track`,
`def_c_root_joint`, duration, motiongraph, and FDB unchanged. Track 114 changed
from a 4.430119 to 44.301189 endpoint range; the MANIS stayed 2,308,703 bytes.
The injected-then-extracted MANIS is byte-identical to the input (SHA-256
`DC21E212FC29E6CB6B2C65DC2D869C17FEC328096F5FF34269884D950124C239`).
The complete 15-file family was installed. **Game result: negative.** The user
observed Patagotitan continuing at its vanilla slow walking pace, so a 10x
`srb` range does not control world displacement.

Test O isolates the scalar `Z Motion Track` at 10x, starting again from vanilla.
The scalar range changed from 4.430119 to 44.301189 while `srb`, root, duration,
motiongraph, and FDB remain untouched. ACL scalar support was added to the raw
patcher using ACL 2.1's documented `scalar_tracks_header`; the size-preserving
patch changes eight bytes. The staged/extracted MANIS is byte-identical at
SHA-256 `3E149E7A6707033DBAC52872A208F6B315E87891A60C8C1E05B3B0568D4097DC`.
The 15-file family was installed. **Game result: partial positive.** In a recorded
Walk-To-Point run, Patagotitan temporarily accelerated, reset backward, and later
teleported forward. Thus `Z Motion Track` is consumed, but it is not the sole
authoritative world transform. The most direct explanation is reconciliation
against the still-vanilla `srb`, root track, or locomotion controller.

Test P changes all three matching clip-side representations coherently: scalar
`Z Motion Track`, `srb` Z, and `def_c_root_joint` Z are each 10x displacement.
Transform ranges scale about their decoded first sample so the root's nonzero
starting pose is preserved; the rejected first draft would incorrectly have
moved root Z from -2.27 to -22.72 at frame zero. The corrected staged/extracted
MANIS is byte-identical at SHA-256
`B0E3AD5C8637D08278B4EC667576E8F098E27AA2D8C62CF64CF59918D719DF70`.
The complete 15-file Test P family was installed. **Game result: fast but
deformed.** The recording shows increased travel speed, but the body repeatedly
stretches/flattens into the ground while the feet remain planted, then recovers.
Scaling `def_c_root_joint` therefore drives skeletal pose against the foot-plant
IK solve; it is not a world-motion authority and must remain vanilla.

Test Q combines only `srb` Z and scalar `Z Motion Track` at 10x, with root and
all other data vanilla. Its staged/extracted MANIS is byte-identical at SHA-256
`C5CD19FEAB29AF607B46F7BD0BED31030C7ADA3B8F084D77FDBEF4C1E8B6C6FC`.
The complete 15-file family was installed; every live hash matched the stage and
no file was zero. **Game result: posture fixed, displacement still unstable.**
The recording shows normal body/foot placement but continued sliding and
discrete teleports. Keeping `srb` coherent with the scalar does not remove
position reconciliation. Sustained world speed is therefore controlled outside
the MANIS by the navigation/locomotion controller or planner lookup.

An in-game ViralSarco test demonstrates why activity-instance coverage matters.
The installed graph had both ordinary preen instances (`StandPreen` and
`StandPreen02`) set to `speed.float=30.0`; VLDinosaurSpawner-triggered preen
visibly accelerated. Only two of 146 `StandIdle01` instances were patched,
leaving 144 at `1.0`, and a Spawner-triggered idle could therefore resolve an
untouched duplicate. This disproves the initial inference that forced Spawner
playback categorically bypasses the motiongraph clock.

A meat-feeder Eat loop also appeared unchanged even though all three ordinary
`Eat02` `AnimationActivity` instances were patched. The feeding state additionally
contains a `RandomAnimationActivity` with `Eat01` weighted 2 and `Eat02` weighted
1; `Eat01` is not represented by an ordinary `AnimationActivity.speed.float`
field. The feeder result therefore does not falsify playback-speed editing: it
may have selected the separately-clocked/random `Eat01`, or another synchronized
interaction path. Its precise active clock remains unproven.

Motiongraphs do not contain audio samples or Wwise banks. Their
`DataStreamProducerActivity` nodes emit named curves with stream name, type,
bone, and location fields; `AudioBlend` streams can drive an audio RTPC or
blend. Custom sound still requires a loaded BNK/event and a compatible consumer.
Within the fixed-topology boundary, an existing data-stream or triggered-activity
slot may be retargeted, but adding an entirely new audio node is not yet safe.

## Surgical topology growth status (2026-08-26)

The earlier conclusion that topology growth is categorically blocked is now
obsolete. Controlled Acrocanthosaurus tests established the following:

- Test D redistributed existing non-null references and ran 5–10 minutes at
  maximum game speed: **game verified**.
- Test E added one net fragment/reference without growing a pool:
  **game verified**.
- Pool tests A3 and A4 grew selected final/structural pools by 16 bytes:
  **game verified**. Pool 0 remains special and is not generalized.
- Tests F/F2 appended an Activity wrapper past a full type-2 page and crashed at
  the same address. Type-2 pools 62–68 show a 16,416-byte page boundary.
- Test F3 appended the wrapper to the final partial type-2 page instead:
  **game verified**.
- Test G independently cloned the wrapper's `ToggledActivityActivityData` into
  the final partial type-3 page. It also moved 1,463 pointers that used the old
  pool end as an empty-array sentinel: **game verified**.
- Test H independently clones the nested 48-byte `AnimationActivity` wrapper
  and its 96-byte `AnimationActivityData`: **game verified**. Because its clip
  remained `Partial_Blank01`, the next test must prove branch execution with an
  unmistakable visible clip. Test H OVL SHA-256:
  `130D514B3A42ABC186F7BB0EB6D4DA262FE820F363B1A7B253FC0E6978CFE620`.

The current invariant is: allocate only in the final partial page for a pool
type, honor the observed page bound, and relocate every fragment whose target
equals the old pool end before using that address as storage. Unknown
`MemPool.num_files` values must not be interpreted as object counts or changed.
Passing cobra reload is only offline evidence; each structural step still needs
an in-game test.

Test K subsequently proved **execution**, not merely acceptance. The four
incoming edges of both stock `StandPreen` wrappers were retargeted to the new
wrapper/data. Changing only the cloned payload's speed from 30.0 to 0.25 changed
the observed head motion by the corresponding amount in game. The motion was
head-only/T-like rather than a proper preen because the data bytes and internal
pointers were cloned from `Partial_Blank01`; retargeting only its MANI string
does not convert the payload's flags, priority, weighting/layer semantics, and
auxiliary data into those of another animation. A correct authored node must
clone the complete donor payload and every internal pointer target.

Test L performed that complete clone from the stock `StandPreen` payload into
the independently allocated data, including its `+0/+56/+72/+88` pointer layout
and additional-data-stream target. Only speed changed, from 1.0 to 0.25. With
Test K's four executable preen edges retained, VLDinosaurSpawner Preen produced
the recognizable full-body animation at quarter speed: **game verified**.

This establishes a safe initial authoring primitive: clone a complete Activity
wrapper and complete type-compatible payload into verified tail-page capacity,
clone every internal fragment target, then retarget an existing executable edge.
It does not yet prove growing a parent's child/reference array or inventing a
new behavior transition from nothing.

The Motiongraph Editor now exposes that narrow primitive as **Clone Activity**.
Select one exact `AnimationActivity` in the States activity tree, copy the full
archive family to Stage / Apply, and clone it with donor speed preserved or an
optional speed override. The writer refuses non-animation activities, unexpected
48/96-byte layouts, missing inbound references, exhausted tail pages, source
overwrite, and different source/output basenames. It relocates old-end sentinels,
updates the archive fragment/size/reservation metadata, and requires both raw and
semantic reload verification.

The generic implementation was first **tool-verified** against the real Acro Test K
family: it cloned global `69/2576` to `69/2672`, cloned data `129/6272` to
`129/6368`, redirected four exact inbound references, added six fragments, moved
1,463 end sentinels, and decoded the clone at speed 0.5. Test M then used the
generalized backend twice in sequence on the two reachable stock Acro
`StandPreen` activities, setting both clones to 8x. The game loaded and the user
observed the accelerated Preen, so the generalized repeated clone writer is now
**game-verified**.
