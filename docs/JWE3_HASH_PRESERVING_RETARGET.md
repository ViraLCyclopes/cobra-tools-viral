# JWE3 hash-preserving asset-family retargeting

JWE3 dinosaur archives cannot currently be renamed safely by changing their ordinary
file hashes. The same DJB2 hash is duplicated across file, root, data, buffer, pool, and
dependency structures, and several arrays are hash-sorted. Rebuilding or permuting those
structures produced archives that cobra could reload but the game rejected.

The supported method uses a private internal alias with all three properties below:

1. It is the same width as the donor prefix.
2. It reaches the same DJB2 state as the donor prefix.
3. It is unique to the mod.

For the game-verified Deinosuchus test:

```text
deinosuchus  ->  sarcmimsaee
djb2(...)    =   1549699023 for both
length       =   11 for both
```

The public package and folder may still use a readable name such as
`SarcoViral_Female`. Only the internal assets and prefab references use the collision
alias, for example `SarcMimsaee_Female`, `SarcMimsaee_Effects`, and animation names such
as `sarcmimsaee$standidle01`.

## Command

Run this from the authoritative cobra-tools fork:

```powershell
python ovl_tool_cmd.py retarget-family `
  "C:\path\to\Deinosuchus_Female.ovl" `
  -o "C:\staging\SarcoViral_Female.ovl" `
  -g "Jurassic World Evolution 3" `
  --from deinosuchus `
  --to sarcmimsaee
```

The command stages the complete family in a temporary directory, then:

- overwrites matching file basenames at their existing offsets;
- overwrites local STATIC pool strings without growing any allocation;
- leaves every file hash, index, dependency, AUX entry, and archive topology unchanged;
- copies every `.ovs.*` companion byte-for-byte under the new OVL stem;
- copies referenced AUX data under the hashed filename required by the new OVL stem;
- reloads and validates the complete staged family before publishing any output file.

It refuses aliases with a different width or DJB2 state. `--force` may replace an
existing output family, but the donor OVL itself can never be the output.

## Prefab/package wiring

The asset package points to the public family path and filename:

```text
...\SarcoViral\Female\SarcoViral_Female
```

The prefab points to the internal alias names contained in that archive:

```text
Effects:     SarcMimsaee_Effects
Layers:      SarcMimsaee_Layers
Model:       SarcMimsaee_Female
MotionGraph: SarcMimsaee_Female
```

This separation is intentional. Renaming the public OVL path does not require its name
to equal the private hash-collision alias.

## Verification level

- **Game-verified:** the Deinosuchus -> `sarcmimsaee` method loaded and ran smoothly.
- **Tool-verified:** the production command preserves the audited invariants, changes
  exactly 221 STATIC strings / 2,431 prefix bytes on the pinned Deinosuchus family, and
  reloads the complete 18-file output.
- Other donors and collision aliases remain tool-verified until tested in game.

This command retargets an existing fixed topology. It does not add or remove assets,
grow motiongraph topology, or make ordinary non-collision renames safe.
