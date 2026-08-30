"""Human-readable state and decision reports for decoded motiongraphs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


CLIP_TYPES = {
    "AnimationActivity": ("mani",),
    "CoordinatedAnimationActivity": ("coordinated_anim", "waiting_anim"),
}
STREAM_TYPES = {"DataStreamProducerActivity"}
CHILD_ACTIVITY_FIELDS = ("activity", "initial_activity", "activities")
FILLER_CLIPS = {"Partial_Blank01", "BindPose"}
UBIQUITY_FRACTION = 0.25
MAX_DECISION_DEPTH = 40


@dataclass(frozen=True)
class ReportStats:
    states: int = 0
    edges: int = 0
    conditions: int = 0
    decision_nodes: int = 0
    decision_blocks: int = 0
    opcodes: int = 0


def build_deref(loader):
    recursion = loader.context.recursion

    def deref(pointer):
        value = getattr(pointer, "data", None)
        if value is not None:
            return value
        pool = getattr(pointer, "target_pool", None)
        offset = getattr(pointer, "target_offset", None)
        if pool is None or offset is None or offset < 0:
            return None
        return recursion.get((pool, offset))

    return deref


def strip_species(name: str) -> str:
    return name.split("$", 1)[1] if "$" in name else name


def collect_subtree(activity, deref, seen=None, depth=0):
    """Yield each activity node once, following shared pointers via the registry."""
    if seen is None:
        seen = set()
    # Some shipped state arrays contain string/sentinel targets in slots that
    # otherwise point at Activity objects. They are not graph nodes.
    if type(activity).__name__ != "Activity" or id(activity) in seen:
        return
    seen.add(id(activity))
    yield activity, depth
    payload = deref(activity.data)
    for attr in ("sub_activities", "other_activities"):
        array = deref(getattr(activity, attr, None)) if hasattr(activity, attr) else None
        if array:
            for reference in array:
                yield from collect_subtree(deref(reference.activity), deref, seen, depth + 1)
    if payload is None:
        return
    for attr in CHILD_ACTIVITY_FIELDS:
        field = getattr(payload, attr, None)
        if field is None:
            continue
        target = deref(field)
        if target is None:
            continue
        if type(target).__name__ == "Activity":
            yield from collect_subtree(target, deref, seen, depth + 1)
            continue
        try:
            entries = list(target)
        except TypeError:
            continue
        for entry in entries:
            for sub_attr in ("activity", "Activity"):
                reference = getattr(entry, sub_attr, None)
                if reference is not None:
                    yield from collect_subtree(deref(reference), deref, seen, depth + 1)
                    break


def activity_clips(activity, deref):
    fields = CLIP_TYPES.get(activity.data_type.data)
    if not fields:
        return []
    payload = deref(activity.data)
    if payload is None:
        return []
    result = []
    for attr in fields:
        value = deref(getattr(payload, attr, None)) if hasattr(payload, attr) else None
        if isinstance(value, str) and value:
            result.append(value)
    return result


def clip_label(clips_by_depth) -> str | None:
    for depth in sorted(clips_by_depth):
        names = [strip_species(name) for name in clips_by_depth[depth]
                 if strip_species(name) not in FILLER_CLIPS]
        if not names:
            continue
        counts = Counter(names)
        if len(counts) == 1:
            return counts.most_common(1)[0][0]
        stems = {match.group(0) for name in names
                 if (match := re.match(r"[A-Z][a-z]+", name))}
        if len(stems) == 1:
            return f"{stems.pop()}*"
        top, count = counts.most_common(1)[0]
        if count > 1 and count >= len(names) // 2:
            return f"{top} +{len(counts) - 1}"
        return " / ".join(sorted(counts)[:2]) + (
            f" +{len(counts) - 2}" if len(counts) > 2 else ""
        )
    return None


def derive_label(clips_by_depth, types, stream_names, state_frequency,
                 total_states: int) -> str:
    label = clip_label(clips_by_depth)
    if label:
        return label
    ceiling = max(1, int(total_states * UBIQUITY_FRACTION))
    distinctive = sorted(
        {name for name in stream_names if name},
        key=lambda name: (state_frequency.get(name, 0), name),
    )
    distinctive = [name for name in distinctive if state_frequency.get(name, 0) <= ceiling]
    if distinctive:
        return "~" + " + ".join(distinctive[:2])
    for depth in sorted(clips_by_depth):
        if clips_by_depth[depth]:
            return strip_species(clips_by_depth[depth][0])
    if types:
        return f"<{types.most_common(1)[0][0]}>"
    return "<empty>"


def analyze_states(loader):
    deref = build_deref(loader)
    activity_addresses = {
        id(value): (int(pool.i), int(offset))
        for (pool, offset), value in loader.context.recursion.items()
        if type(value).__name__ == "Activity"
    }
    header = loader.header
    state_array = deref(header.state_output_entries)
    references = deref(state_array.states)
    states = [deref(reference.state) for reference in references]
    index_of = {id(state): index for index, state in enumerate(states) if state is not None}
    inbound = defaultdict(list)
    for value in loader.context.recursion.values():
        if type(value).__name__ != "MRFMember1":
            continue
        method = deref(value.lua_method)
        target = deref(value.ptr_0)
        if target is not None and type(target).__name__ == "State":
            inbound[id(target)].append(method)

    rows = []
    for index, state in enumerate(states):
        if state is None:
            rows.append(None)
            continue
        clips_by_depth, types, streams = defaultdict(list), Counter(), []
        total_nodes, seen, addresses = 0, set(), set()
        for reference in (deref(state.activities) or []):
            for node, depth in collect_subtree(deref(reference.activity), deref, seen):
                total_nodes += 1
                address = activity_addresses.get(id(node))
                if address is not None:
                    addresses.add(address)
                activity_type = node.data_type.data
                types[activity_type] += 1
                clips_by_depth[depth].extend(activity_clips(node, deref))
                if activity_type in STREAM_TYPES:
                    payload = deref(node.data)
                    if payload is not None:
                        streams.append({
                            "name": deref(getattr(payload, "ds_name", None)),
                            "type": deref(getattr(payload, "type", None)),
                        })
        edges = []
        for transition in (deref(state.array_2) or []):
            targets = []
            target_array = deref(transition.states.states)
            if target_array:
                for reference in target_array:
                    target = deref(reference.state)
                    targets.append(index_of.get(id(target)))
            conditions = []
            for record in (deref(transition.another_mrf_reference_2) or []):
                conditions.append({
                    "trigger": deref(record.trigger),
                    "tier": int(record.tier),
                    "num_activities": int(record.num_activities),
                    "activity_flags": int(record.activity_flags),
                    "curve_length": float(record.curve_length),
                })
            edges.append({"targets": targets, "conditions": conditions})
        all_clips = [name for names in clips_by_depth.values() for name in names]
        rows.append({
            "index": index, "label": None,
            "_clips_by_depth": {depth: list(names) for depth, names in clips_by_depth.items()},
            "_types": types, "activity_nodes": total_nodes,
            "clips": sorted({strip_species(name) for name in all_clips}),
            "types": dict(types), "streams": streams, "edges": edges,
            "inbound_decisions": Counter(inbound.get(id(state), [])),
            "activity_addresses": sorted(addresses),
        })
    valid = [row for row in rows if row]
    frequency = Counter()
    for row in valid:
        for name in {stream["name"] for stream in row["streams"] if stream["name"]}:
            frequency[name] += 1
    for row in valid:
        row["label"] = derive_label(
            row.pop("_clips_by_depth"), row.pop("_types"),
            [stream["name"] for stream in row["streams"]], frequency, len(valid),
        )
    return deref, states, rows


def build_activity_tree(loader, state_index: int, max_nodes: int = 5000):
    """Return the nested activity composition for one anonymous state."""
    # Do not call analyze_states here: it walks every activity subtree for all
    # states and makes interactive selection unnecessarily slow.
    deref = build_deref(loader)
    activity_addresses = {
        id(value): (int(pool.i), int(offset))
        for (pool, offset), value in loader.context.recursion.items()
        if type(value).__name__ == "Activity"
    }
    state_array = deref(loader.header.state_output_entries)
    references = deref(state_array.states)
    states = [deref(reference.state) for reference in references]
    if state_index < 0 or state_index >= len(states) or states[state_index] is None:
        raise ValueError(f"State index is not valid: {state_index}")
    state = states[state_index]
    seen = {}
    count = 0

    def activity_name(activity, payload):
        for owner in (activity, payload):
            if owner is None:
                continue
            for attr in ("name_b", "name", "mani", "activity_name", "variable", "ds_name"):
                value = deref(getattr(owner, attr, None))
                if isinstance(value, str) and value:
                    return value
        return None

    def source_address(pointer):
        pool = getattr(pointer, "src_pool", None)
        offset = getattr(pointer, "io_start", None)
        if pool is None or offset is None or int(offset) < 0:
            return None
        return int(pool.i), int(offset)

    def children_of(activity, payload):
        children = []
        for attr in ("sub_activities", "other_activities"):
            array = deref(getattr(activity, attr, None))
            for index, reference in enumerate(array or []):
                pointer = getattr(reference, "activity", None)
                children.append((f"{attr}[{index}]", deref(pointer), pointer))
        if payload is None:
            return children
        for attr in CHILD_ACTIVITY_FIELDS:
            field = getattr(payload, attr, None)
            if field is None:
                continue
            target = deref(field)
            if type(target).__name__ == "Activity":
                children.append((attr, target, field))
                continue
            try:
                entries = list(target or [])
            except TypeError:
                continue
            for index, entry in enumerate(entries):
                reference = getattr(entry, "activity", None)
                if reference is None:
                    reference = getattr(entry, "Activity", None)
                child = deref(reference)
                if child is not None:
                    children.append((f"{attr}[{index}]", child, reference))
        return children

    def render(activity, relationship, inbound_pointer=None):
        nonlocal count
        if type(activity).__name__ != "Activity":
            return None
        identity = id(activity)
        if identity in seen:
            return {
                "relationship": relationship, "activity_type": "shared reference",
                "label": f"see node {seen[identity]}", "clips": [], "children": [],
                "shared": True, "target_node": seen[identity],
                "address": activity_addresses.get(identity),
                "inbound_source": source_address(inbound_pointer),
            }
        if count >= max_nodes:
            return {
                "relationship": relationship, "activity_type": "limit",
                "label": f"activity tree limited to {max_nodes} unique nodes",
                "clips": [], "children": [], "shared": True,
            }
        count += 1
        number = count
        seen[identity] = number
        payload = deref(activity.data)
        clips = activity_clips(activity, deref)
        node = {
            "relationship": relationship,
            "node": number,
            "address": activity_addresses.get(identity),
            "inbound_source": source_address(inbound_pointer),
            "activity_type": activity.data_type.data,
            "label": activity_name(activity, payload),
            "clips": clips,
            "children": [],
            "shared": False,
        }
        for child_relationship, child, pointer in children_of(activity, payload):
            rendered = render(child, child_relationship, pointer)
            if rendered is not None:
                node["children"].append(rendered)
        return node

    roots = []
    for index, reference in enumerate(deref(state.activities) or []):
        root = render(
            deref(reference.activity), f"state.activities[{index}]", reference.activity
        )
        if root is not None:
            roots.append(root)
    return roots, count


def build_state_report(loader):
    deref, states, rows = analyze_states(loader)
    valid = [row for row in rows if row]
    edge_count = sum(len(edge["targets"]) for row in valid for edge in row["edges"])
    condition_count = sum(len(edge["conditions"]) for row in valid for edge in row["edges"])
    stream_names = Counter(stream["name"] for row in valid for stream in row["streams"])
    stream_types = Counter(stream["type"] for row in valid for stream in row["streams"])
    routed = sum(sum(row["inbound_decisions"].values()) for row in valid)
    lines = [
        f"# Motiongraph state report — {loader.name}", "",
        f"- states: {len(valid)}", f"- state-to-state edges: {edge_count}",
        f"- transition condition records: {condition_count}",
        f"- decision nodes routing into states: {routed}", "",
        "Labels are derived because states are anonymous and positional. A plain label comes "
        "from the shallowest non-placeholder clip; `~Signal` identifies a placeholder state "
        "by a distinctive data-stream signal.", "", "## Data stream categories", "",
        "| category | count |", "|---|---:|",
    ]
    for name, count in stream_types.most_common():
        lines.append(f"| {name or '(none)'} | {count} |")
    lines.extend(["", "Most frequent signals: " + ", ".join(
        f"`{name}` x{count}" for name, count in stream_names.most_common(10) if name
    ), "", "## States", ""])
    for row in valid:
        lines.extend([f"### [{row['index']}] {row['label']}", "",
                      f"- activity nodes: {row['activity_nodes']}"])
        if row["types"]:
            lines.append("- composition: " + ", ".join(
                f"{name} x{count}" for name, count in sorted(
                    row["types"].items(), key=lambda item: -item[1]
                )
            ))
        if row["clips"]:
            shown = row["clips"][:12]
            suffix = "" if len(row["clips"]) <= 12 else f" (+{len(row['clips']) - 12} more)"
            lines.append("- clips: " + ", ".join(f"`{name}`" for name in shown) + suffix)
        if row["streams"]:
            names = sorted({stream["name"] for stream in row["streams"] if stream["name"]})
            if names:
                lines.append("- streams: " + ", ".join(f"`{name}`" for name in names[:10]))
        if row["inbound_decisions"]:
            lines.append("- entered via: " + ", ".join(
                f"{method.split('.')[-1]} x{count}"
                for method, count in row["inbound_decisions"].most_common()
            ))
        for edge in row["edges"]:
            targets = [target for target in edge["targets"] if target is not None]
            if not targets:
                continue
            tiers = sorted({condition["tier"] for condition in edge["conditions"]})
            triggers = sorted({condition["trigger"] for condition in edge["conditions"]
                               if condition["trigger"]})
            details = f" tiers={tiers}" if tiers else ""
            details += f" triggers={triggers}" if triggers else ""
            lines.append(f"- -> {targets}{details}")
        lines.append("")
    payload = [
        {**row, "inbound_decisions": dict(row["inbound_decisions"])} if row else None
        for row in rows
    ]
    return "\n".join(lines), payload, ReportStats(
        states=len(valid), edges=edge_count, conditions=condition_count,
    )


def _short(method):
    return method.split(".")[-1] if method else "?"


def _param_text(something, deref):
    if something is None:
        return None
    inner = deref(getattr(something, "ptr", None))
    if inner is None:
        return None
    kind = type(inner).__name__
    if kind == "MotiongraphVar":
        return f"{deref(inner.var_name)} = {deref(inner.target_name)!r}"
    if kind == "MotiongraphResultParam":
        return f"{deref(inner.field)} = {int(inner.value)} (type {int(inner.type)})"
    if kind == "MotiongraphRangeParams":
        return (f"{deref(inner.minimum.name)} = {int(inner.minimum.value)}, "
                f"{deref(inner.maximum.name)} = {int(inner.maximum.value)}")
    return kind


def _node_fields(node, deref):
    variables = deref(node.motiongraph_vars)
    array = deref(getattr(variables, "ptr", None)) if variables is not None else None
    result = []
    for entry in (array or []):
        name = deref(getattr(entry, "var_name", None))
        value = deref(getattr(entry, "target_name", None))
        if name:
            result.append(f"{name} = {value!r}")
    return result


def build_decision_graph(loader, state_rows=None):
    """Return decision opcodes and result branches as graph-ready data."""
    deref = build_deref(loader)
    state_array = deref(loader.header.state_output_entries)
    references = deref(state_array.states)
    states = [deref(reference.state) for reference in references]
    state_index = {id(state): index for index, state in enumerate(states) if state is not None}
    state_labels = {
        row["index"]: row["label"] for row in (state_rows or []) if row is not None
    }
    nodes, edges, roots = [], [], []
    node_ids = {}

    def visit(node):
        if node is None or type(node).__name__ != "MRFMember1":
            return None
        identity = id(node)
        if identity in node_ids:
            return node_ids[identity]
        index = len(nodes)
        node_ids[identity] = index
        method = deref(node.lua_method)
        target = deref(node.ptr_0)
        target_index = state_index.get(id(target)) if target is not None else None
        nodes.append({
            "index": index,
            "opcode": method,
            "fields": _node_fields(node, deref),
            "target_state": target_index,
            "target_label": state_labels.get(target_index),
        })
        for result_index, child in enumerate(deref(node.children) or [], start=1):
            parameters = _param_text(deref(child.ptr_1), deref)
            child_index = visit(deref(child.m_r_f_member))
            if child_index is not None:
                edges.append({
                    "source": index, "target": child_index,
                    "result": result_index, "parameters": parameters,
                })
        return index

    root = visit(deref(loader.header.m_r_f_member_1))
    if root is not None:
        roots.append({"node": root, "label": "Root decision tree"})
    for state_number, state in enumerate(states):
        if state is None:
            continue
        for transition in (deref(state.array_2) or []):
            for record in (deref(transition.another_mrf_reference_2) or []):
                instructions = deref(record.decision_instructions)
                if instructions is None:
                    continue
                try:
                    items = list(instructions)
                except TypeError:
                    items = [instructions]
                for item in items:
                    instruction = visit(item)
                    if instruction is not None:
                        roots.append({
                            "node": instruction,
                            "label": f"STATE[{state_number}] tier {int(record.tier)}",
                        })
    return {"nodes": nodes, "edges": edges, "roots": roots}


class _DecisionRenderer:
    def __init__(self, deref, state_index, state_labels):
        self.deref = deref
        self.state_index = state_index
        self.state_labels = state_labels
        self.lines = []
        self.methods = Counter()
        self.node_ids = {}

    def render(self, node, indent=0, path=None):
        if node is None or type(node).__name__ != "MRFMember1":
            return
        path = path or set()
        pad = "    " * indent
        if id(node) in path or indent > MAX_DECISION_DEPTH:
            self.lines.append(pad + "... (cycle)")
            return
        if id(node) in self.node_ids:
            self.lines.append(
                f"{pad}{_short(self.deref(node.lua_method))} -> see node #{self.node_ids[id(node)]}"
            )
            return
        number = len(self.node_ids) + 1
        self.node_ids[id(node)] = number
        path = path | {id(node)}
        method = self.deref(node.lua_method)
        self.methods[method] += 1
        head = f"#{number} {_short(method)}"
        fields = _node_fields(node, self.deref)
        if fields:
            head += " " + ", ".join(fields)
        target = self.deref(node.ptr_0)
        if target is not None and type(target).__name__ == "State":
            index = self.state_index.get(id(target))
            if index is not None:
                head += f"  ->  STATE[{index}] {self.state_labels.get(index, '?')}"
        self.lines.append(pad + head)
        for index, child in enumerate(self.deref(node.children) or [], start=1):
            params = _param_text(self.deref(child.ptr_1), self.deref)
            body = self.deref(child.m_r_f_member)
            if body is None and not params:
                continue
            tag = f"{pad}    result {index}" + (f" [{params}]" if params else "")
            self.lines.append(tag + ":")
            self.render(body, indent + 2, path)


def build_decision_report(loader):
    deref, states, rows = analyze_states(loader)
    state_index = {id(state): index for index, state in enumerate(states) if state is not None}
    labels = {row["index"]: row["label"] for row in rows if row}
    renderer = _DecisionRenderer(deref, state_index, labels)
    renderer.lines.extend(["## Root decision tree", "", "```"])
    renderer.render(deref(loader.header.m_r_f_member_1))
    renderer.lines.extend(["```", "", "## Transition decision instructions", ""])
    blocks = 0
    for index, state in enumerate(states):
        if state is None:
            continue
        for transition in (deref(state.array_2) or []):
            for record in (deref(transition.another_mrf_reference_2) or []):
                instructions = deref(record.decision_instructions)
                if instructions is None:
                    continue
                items = [item for item in (
                    list(instructions) if hasattr(instructions, "__iter__") else [instructions]
                ) if item is not None]
                if not items:
                    continue
                trigger = deref(record.trigger)
                title = f"### from STATE[{index}] {labels.get(index, '?')} — tier {int(record.tier)}"
                if trigger:
                    title += f", trigger `{trigger}`"
                renderer.lines.extend([title, "", "```"])
                for item in items:
                    renderer.render(item)
                renderer.lines.extend(["```", ""])
                blocks += 1
    header = [
        f"# Motiongraph decision report — {loader.name}", "",
        f"- decision nodes rendered: {len(renderer.node_ids)}",
        f"- transition instruction blocks: {blocks}", "",
        "A node switches on a named variable. `result N` is its Nth binary result "
        "parameter, and output nodes point to the anonymous states labelled by the state report.",
        "", "## Opcodes", "", "| opcode | count |", "|---|---:|",
    ]
    for method, count in renderer.methods.most_common():
        header.append(f"| `{method}` | {count} |")
    header.append("")
    report = "\n".join(header + renderer.lines)
    return report, ReportStats(
        states=sum(state is not None for state in states),
        decision_nodes=len(renderer.node_ids), decision_blocks=blocks,
        opcodes=len(renderer.methods),
    )
