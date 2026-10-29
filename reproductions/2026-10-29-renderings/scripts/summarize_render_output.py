#!/usr/bin/env python3
"""Validate the six audited five-episode rendering batches and build an index.

Only the exact current Stage-I/II HA-RMAPPO summary paths are read. Archived or
unverified images are never discovered through globbing. No simulation, policy
inference, PDF creation, or modification of the source renderings occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCALES = {"2v1": (2, 1), "5v2": (5, 2), "7v3": (7, 3)}
FORMATS = ("png", "gif", "pdf", "csv")
TABLE_FIELDS = (
    "stage", "scale", "episode", "seed", "status", "verified_defense_success",
    "returned_defense_success", "capture_distances_valid",
    "all_captured", "captured_count", "captured_ids", "attack_success", "timeout",
    "terminal_step", "duration_seconds", "min_zone_clearance", "min_capture_distance",
    "max_capture_distance", "png", "gif", "pdf", "csv", "source_summary",
)


def repository_path(value, repository_root=REPO_ROOT):
    """Resolve fresh-clone references while accepting absolute runtime paths."""
    path = Path(value)
    resolved = (path if path.is_absolute() else repository_root / path).resolve()
    require(resolved.is_relative_to(repository_root.resolve()),
            "A recorded artifact path escapes the repository root")
    return resolved


def repository_reference(value):
    return repository_path(value).relative_to(REPO_ROOT).as_posix()


def normalized_metadata(value):
    if isinstance(value, dict):
        return {key: normalized_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_metadata(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value).resolve()
        if path.is_relative_to(REPO_ROOT):
            return path.relative_to(REPO_ROOT).as_posix()
    return value


def verify_public_text(content):
    require(re.search(r"[\u3400-\u9fff]", content) is None,
            "Publication text must be in English")
    require(re.search(r"/(?:Users|home)/[^/\s]+", content) is None,
            "Publication text contains a local user path")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value, context):
    number = float(value)
    require(math.isfinite(number), f"Nonfinite number: {context}")
    return number


def integer(value, context):
    number = finite(value, context)
    require(number == int(number), f"Nonintegral number: {context}")
    return int(number)


def boolean(value, context):
    require(isinstance(value, bool), f"Expected JSON boolean: {context}")
    return value


def close(actual, expected, context, tolerance=1e-7):
    require(math.isclose(finite(actual, context), finite(expected, context),
                         rel_tol=tolerance, abs_tol=tolerance),
            f"Numeric mismatch {context}: {actual} != {expected}")


def coordinate(value, context):
    require(isinstance(value, list) and len(value) == 2, f"Expected 2D position: {context}")
    return [finite(component, context) for component in value]


def norm(vector):
    return math.hypot(*vector)


def segment_distance(start, end, center):
    direction = [end[i] - start[i] for i in range(2)]
    length_squared = sum(value * value for value in direction)
    projection = sum((center[i] - start[i]) * direction[i] for i in range(2)) / max(length_squared, 1e-30)
    projection = max(0.0, min(1.0, projection))
    return norm([start[i] + projection * direction[i] - center[i] for i in range(2)])


def inspect_csv(path, episode, defenders, intruders, cfg):
    terminal_step = integer(episode["terminal_step"], str(path))
    require(1 <= terminal_step <= cfg["episode_length"], f"Invalid terminal step: {path}")
    require(integer(episode["plot_frame_count"], str(path)) == terminal_step,
            f"Native pre-step frame count differs from terminal step: {path}")
    positions = {}
    terminal, terminal_sticky = {}, set()
    required = {"episode", "step", "category", "entity_type", "entity_index", "x", "y",
                "time", "captured", "frame_phase", "sticky_breach"}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(required <= set(reader.fieldnames or []), f"Missing audited CSV columns: {path}")
        for row in reader:
            if row["category"] not in ("agent", "terminal_agent"):
                continue
            require(integer(row["episode"], str(path)) == episode["episode"],
                    f"CSV episode mismatch: {path}")
            step = integer(row["step"], str(path))
            close(row["time"], step * 0.1, f"{path}: physical time")
            group = row["entity_type"]
            require(group in ("defender", "intruder"), f"Unknown CSV agent group: {path}")
            index = integer(row["entity_index"], str(path))
            count = defenders if group == "defender" else intruders
            require(0 <= index < count, f"Invalid CSV agent index: {path}")
            position = [finite(row["x"], str(path)), finite(row["y"], str(path))]
            key = (step, group, index)
            require(key not in positions, f"Duplicate agent sample in CSV: {path}: {key}")
            positions[key] = position
            if row["category"] == "terminal_agent":
                require(step == terminal_step and row["frame_phase"] == "terminal_post_step",
                        f"Invalid CSV terminal row: {path}")
                terminal[(group, index)] = (position, row["captured"])
                terminal_sticky.add(integer(row["sticky_breach"], str(path)))
            else:
                require(0 <= step < terminal_step and row["frame_phase"] == "native_pre_step",
                        f"Invalid native CSV frame: {path}")
    expected_count = (terminal_step + 1) * (defenders + intruders)
    require(len(positions) == expected_count, f"Incomplete CSV trajectory: {path}")
    for step in range(terminal_step + 1):
        for group, count in (("defender", defenders), ("intruder", intruders)):
            for index in range(count):
                require((step, group, index) in positions, f"Missing CSV frame/agent: {path}")
    require(len(terminal) == defenders + intruders, f"Incomplete CSV terminal state: {path}")
    final = episode["final_positions"]
    require(len(final) == defenders + intruders, f"Summary final agent count differs: {path}")
    for offset, (group, count) in ((0, ("defender", defenders)), (defenders, ("intruder", intruders))):
        for index in range(count):
            expected = coordinate(final[offset + index], str(path))
            observed, captured = terminal[(group, index)]
            for axis in range(2):
                close(observed[axis], expected[axis], f"{path}: final {group} {index}")
            if group == "intruder":
                require(integer(captured, str(path)) == int(index in episode["captured_ids"]),
                        f"CSV terminal captured flag differs: {path}")
    require(terminal_sticky == {int(episode["cumulative_sticky_breach"])},
            f"CSV terminal sticky breach differs: {path}")
    for event in episode["capture_events"]:
        step = integer(event["step"], str(path))
        for group, index, field in (("intruder", event["intruder_index"], "intruder_position"),
                                    ("defender", event["defender_index"], "defender_position")):
            for axis in range(2):
                close(positions[(step, group, index)][axis], event[field][axis],
                      f"{path}: capture event position")
    center, radius = [cfg["protected_cx"], cfg["protected_cy"]], cfg["protected_r"]
    sampled, segments = [], []
    for index in range(intruders):
        track = [positions[(step, "intruder", index)] for step in range(terminal_step + 1)]
        sampled.append(min(norm([point[i] - center[i] for i in range(2)]) for point in track) - radius)
        segments.append(min(segment_distance(start, end, center) for start, end in zip(track, track[1:])) - radius)
    require(len(episode["min_sampled_clearance_by_intruder"]) == intruders,
            f"Missing per-intruder sample clearance: {path}")
    require(len(episode["min_segment_clearance_by_intruder"]) == intruders,
            f"Missing per-intruder segment clearance: {path}")
    for index in range(intruders):
        close(sampled[index], episode["min_sampled_clearance_by_intruder"][index],
              f"{path}: sampled clearance {index}")
        close(segments[index], episode["min_segment_clearance_by_intruder"][index],
              f"{path}: segment clearance {index}")
    require(any(value <= 0 for value in sampled) == episode["sampled_position_breach"],
            f"CSV sampled breach classification differs: {path}")
    require(any(value <= 0 for value in segments) == episode["interpolated_segment_breach"],
            f"CSV segment breach classification differs: {path}")
    return {"trajectory_rows": expected_count, "terminal_agent_rows": len(terminal),
            "sampled_clearance_by_intruder": sampled, "segment_clearance_by_intruder": segments,
            "terminal_state_matches": True, "classification_matches": True}


def inspect_episode(episode, stage, scale, cfg, render_root):
    defenders, intruders = SCALES[scale]
    prefix = f"stage{stage}/{scale}/episode{episode.get('episode')}"
    number = integer(episode["episode"], prefix)
    seed = integer(episode["seed"], prefix)
    captured_ids = episode["captured_ids"]
    require(isinstance(captured_ids, list) and captured_ids == sorted(set(captured_ids)),
            f"Captured IDs must be sorted and unique: {prefix}")
    require(all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < intruders
                for index in captured_ids), f"Invalid captured IDs: {prefix}")
    all_captured = boolean(episode["all_captured"], prefix)
    require(all_captured == (captured_ids == list(range(intruders))), f"All-captured mismatch: {prefix}")
    breach_keys = ("cumulative_sticky_breach", "sampled_position_breach", "interpolated_segment_breach")
    breach_flags = [boolean(episode[key], f"{prefix}/{key}") for key in breach_keys]
    breached = any(breach_flags)
    attack_success = boolean(episode["attack_success"], prefix)
    require(attack_success == breached, f"Attack success differs from cumulative breach: {prefix}")
    success = boolean(episode["verified_defense_success"], prefix)
    returned_success = finite(episode["native_stats"]["defense_success"], prefix) > 0.5
    require(boolean(episode["returned_defense_success"], prefix) == returned_success,
            f"Native/returned success flags differ: {prefix}")
    timeout = boolean(episode["timeout"], prefix)
    terminal_step = integer(episode["terminal_step"], prefix)
    require(timeout == (not success and not breached and terminal_step >= cfg["episode_length"]),
            f"Timeout classification differs: {prefix}")
    events = episode["capture_events"]
    require(isinstance(events, list), f"Capture events must be a list: {prefix}")
    event_ids, event_distances = [], []
    for event in events:
        index = integer(event["intruder_index"], prefix)
        defender_index = integer(event["defender_index"], prefix)
        require(0 <= defender_index < defenders, f"Invalid capture defender: {prefix}")
        event_ids.append(index)
        distance = finite(event["distance"], prefix)
        require(0 <= distance <= cfg["capture_r"] + 1e-6, f"Capture outside radius: {prefix}")
        intruder_position = coordinate(event["intruder_position"], prefix)
        defender_position = coordinate(event["defender_position"], prefix)
        close(norm([intruder_position[i] - defender_position[i] for i in range(2)]), distance,
              prefix + ": capture event geometry")
        event_step = integer(event["step"], prefix)
        require(1 <= event_step <= terminal_step, f"Capture event outside episode: {prefix}")
        close(event["time"], event_step * 0.1, prefix + ": capture time")
        event_distances.append(distance)
    require(sorted(event_ids) == captured_ids, f"Capture events do not match captured IDs: {prefix}")
    capture_distances_valid = boolean(episode["capture_distances_valid"], prefix)
    require(capture_distances_valid == (len(events) == intruders
                                       and all(distance <= cfg["capture_r"] + 1e-7 for distance in event_distances)),
            f"Capture-distance validity flag differs: {prefix}")
    require(success == (returned_success and all_captured and capture_distances_valid and not breached),
            f"Defense success classification differs: {prefix}")
    files, hashes = {}, {}
    for extension in FORMATS:
        value = episode["files"].get(extension)
        require(isinstance(value, str), f"Missing {extension} path: {prefix}")
        path = repository_path(value)
        expected = render_root / f"stage{stage}/rmappo_{scale}_stage{stage}_ep{number:02d}.{extension}"
        require(path.resolve() == expected.resolve(), f"Unexpected output filename/location: {path}")
        require(path.is_file() and path.stat().st_size > 0, f"Missing/empty output file: {path}")
        with path.open("rb") as stream:
            header = stream.read(8)
            if extension == "png":
                require(header == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")
            elif extension == "gif":
                require(header[:6] in (b"GIF87a", b"GIF89a"), f"Invalid GIF signature: {path}")
            elif extension == "pdf":
                require(header.startswith(b"%PDF-"), f"Invalid PDF header: {path}")
                stream.seek(max(0, path.stat().st_size - 1024))
                require(b"%%EOF" in stream.read(), f"Missing PDF EOF marker: {path}")
        files[extension] = repository_reference(path)
        hashes[extension] = {"path": repository_reference(path), "sha256": digest(path), "bytes": path.stat().st_size}
        require(episode["file_sha256"].get(extension) == hashes[extension]["sha256"],
                f"Output file differs from rendering summary hash: {path}")
    csv_validation = inspect_csv(repository_path(files["csv"]), episode, defenders, intruders, cfg)
    status = "success" if success else "breach" if breached else "timeout" if timeout else "unresolved"
    row = {"stage": stage, "scale": scale, "episode": number, "seed": seed, "status": status,
           "verified_defense_success": success, "all_captured": all_captured,
           "returned_defense_success": returned_success, "capture_distances_valid": capture_distances_valid,
           "captured_count": len(captured_ids), "captured_ids": json.dumps(captured_ids),
           "attack_success": attack_success, "timeout": timeout, "terminal_step": terminal_step,
           "duration_seconds": terminal_step * 0.1,
           "min_zone_clearance": min(csv_validation["segment_clearance_by_intruder"]),
           "min_capture_distance": min(event_distances) if event_distances else "",
           "max_capture_distance": max(event_distances) if event_distances else "", **files}
    return row, {"episode": number, "seed": seed, "status": status,
                 "files": hashes, "csv_validation": csv_validation,
                 "classification_validation": True, "capture_event_validation": True}


def inspect_configuration(render_root, stage, scale, expected_episodes, checkpoint_root):
    path = render_root / f"stage{stage}/rmappo_{scale}_stage{stage}_summary.json"
    summary = json.loads(path.read_text())
    metadata = summary["metadata"]
    require(metadata.get("kind") == "native_renderer_independent_seeded_episodes",
            f"Not a current audited rendering summary: {path}")
    require(metadata.get("stage") == stage and metadata.get("requested_episodes") == expected_episodes,
            f"Summary batch metadata differs: {path}")
    require(metadata["worldparams"].get("config_matches_actual_world") is True,
            f"Missing actual-world parameter validation: {path}")
    roles = ("defender_actor", "defender_critic", "intruder_actor", "intruder_critic")
    require(set(metadata["checkpoint_hashes"]) == set(roles), f"Expected four loaded checkpoints: {path}")
    for role in roles:
        checkpoint = metadata["checkpoint_hashes"][role]
        require(all(checkpoint.get(key) is True for key in
                    ("strict", "all_keys_match", "all_shapes_match", "all_tensor_values_equal")),
                f"Unverified/partial checkpoint load: {path}: {role}")
        checkpoint_path = Path(checkpoint["path"])
        if checkpoint_path.is_absolute():
            checkpoint_path = repository_path(checkpoint_path).relative_to(REPO_ROOT)
        require(digest(repository_path(checkpoint_path, checkpoint_root)) == checkpoint["sha256"],
                f"Checkpoint changed since rendering: {path}: {role}")
    cfg = metadata["config"]
    defenders, intruders = SCALES[scale]
    expected = {"algorithm_name": "rmappo", "num_defenders": defenders,
                "num_intruders": intruders, "num_agents": defenders + intruders,
                "scenario_name": f"protected_zone_stage{stage}", "episode_length": 300,
                "protected_cx": 0.0, "protected_cy": 0.0, "world_r": 5.0,
                "protected_r": 0.5, "capture_r": 0.2,
                "defender_max_speed": 1.0, "intruder_max_speed": 1.0}
    for key, value in expected.items():
        require(cfg.get(key) == value, f"Rendering config mismatch {path}: {key}={cfg.get(key)!r}")
    episodes = summary["episodes"]
    require(isinstance(episodes, list) and len(episodes) == expected_episodes,
            f"Expected exactly {expected_episodes} episodes: {path}")
    require([episode["episode"] for episode in episodes] == list(range(1, expected_episodes + 1)),
            f"Expected ordered episode IDs 1..{expected_episodes}: {path}")
    require([episode["seed"] for episode in episodes] ==
            list(range(metadata["base_seed"], metadata["base_seed"] + expected_episodes)),
            f"Episode seeds differ from declared consecutive panel: {path}")
    rows, audit = [], []
    for episode in episodes:
        row, validation = inspect_episode(episode, stage, scale, cfg, render_root)
        row["source_summary"] = repository_reference(path)
        rows.append(row)
        audit.append(validation)
    successes = [row["episode"] for row in rows if row["verified_defense_success"]]
    require(summary["success_episodes"] == successes, f"Success episode index differs: {path}")
    return rows, {"stage": stage, "scale": scale, "source_summary": repository_reference(path),
                  "source_summary_sha256": digest(path), "metadata": normalized_metadata(metadata),
                  "episode_count": expected_episodes, "success_episodes": successes,
                  "breach_episodes": [row["episode"] for row in rows if row["attack_success"]],
                  "timeout_episodes": [row["episode"] for row in rows if row["timeout"]],
                  "episodes": audit}


def local_url(path, root):
    return quote(repository_path(path).relative_to(root).as_posix(), safe="/")


def build_html(rows, render_root):
    sections = []
    labels = {"success": "Success", "breach": "Breach",
              "timeout": "Timeout", "unresolved": "Unresolved"}
    for stage in (1, 2):
        for scale, (_, intruders) in SCALES.items():
            selected = [row for row in rows if row["stage"] == stage and row["scale"] == scale]
            successes = [str(row["episode"]) for row in selected if row["verified_defense_success"]]
            cards = []
            for row in selected:
                links = " · ".join(f'<a href="{local_url(row[extension], render_root)}">{extension.upper()}</a>'
                                   for extension in FORMATS)
                cards.append(f'''<article class="card {row['status']}">
<h3>Episode {row['episode']} <span>{labels[row['status']]}</span></h3>
<a href="{local_url(row['png'], render_root)}"><img src="{local_url(row['png'], render_root)}" alt="Stage {stage}, {scale}, episode {row['episode']}: {labels[row['status']]}" loading="lazy"></a>
<p>Seed {row['seed']} · Captured {row['captured_count']}/{intruders} · {row['duration_seconds']:.1f} s</p>
<p class="formats">{links}</p></article>''')
            sections.append(f'''<section><h2>Stage {stage} · {scale}</h2>
<p>Successful episodes: {', '.join(successes) if successes else 'none'}. {len(selected)} predetermined-seed episodes.</p>
<div class="cards">{''.join(cards)}</div></section>''')
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HA-RMAPPO rendering outputs</title>
<style>body{{font-family:system-ui,sans-serif;background:#f5f7fa;color:#1b2534;margin:0 auto;padding:28px;max-width:1600px}}h1{{font-size:28px}}h2{{margin:30px 0 8px}}.intro{{max-width:1050px;line-height:1.7}}a{{color:#145a90}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}}.card{{background:white;border:1px solid #d7dee6;border-top:5px solid #6b7280;border-radius:7px;padding:12px;min-width:0}}.card.success{{border-top-color:#16824c}}.card.breach{{border-top-color:#bc452e}}.card.timeout{{border-top-color:#b07b13}}h3{{font-size:15px;margin:0 0 10px}}h3 span{{float:right;font-size:12px;font-weight:500}}img{{display:block;width:100%;height:auto}}.card p{{font-size:13px;margin:10px 0 0}}.formats a{{display:inline-block;padding:4px 0}}footer{{margin:28px 0;color:#556274;font-size:13px}}</style>
<h1>HA-RMAPPO · two stages and three team sizes</h1>
<p class="intro">Each configuration runs {len(rows) // 6} consecutive predetermined-seed episodes, retaining every success, breach, timeout and unresolved outcome. Verified success requires the native success flag, capture of every intruder within the capture radius, and no sticky, sampled-position or interpolated-segment breach. Capture distances and CSV terminal states are checked independently. Native callback boundary inconsistencies remain unverified. These {len(rows)} fixed-checkpoint episodes illustrate policy behavior and are not an estimate of the paper's defense success rate (DSR).</p>
<p><a href="success_summary.csv">Per-episode results CSV</a> · <a href="render_manifest.json">Provenance, file hashes and validation manifest</a></p>
{''.join(sections)}
<footer>Only artifacts named by the six current audited summaries are shown; historical _archive files are not scanned. Dynamics and rewards retain the repository implementation. Differences from the paper are recorded in each configuration's provenance metadata.</footer></html>\n'''


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, default=PACKAGE_ROOT / "replay_output")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT,
                        help="Repository root containing results/MPE checkpoints; override for read-only staging validation")
    parser.add_argument("--expected-episodes", type=int, default=5,
                        help="Required number of episodes in every configuration (default: 5)")
    parser.add_argument("--validate-only", action="store_true", help="Validate the complete package without writing index/CSV/manifest")
    options = parser.parse_args()
    require(options.expected_episodes >= 1, "--expected-episodes must be positive")
    render_root = options.render_root.resolve()
    require(render_root.is_relative_to(REPO_ROOT), "--render-root must remain inside this repository")
    checkpoint_root = options.checkpoint_root.resolve()
    rows, configurations = [], []
    for stage in (1, 2):
        for scale in SCALES:
            result, audit = inspect_configuration(render_root, stage, scale, options.expected_episodes, checkpoint_root)
            rows.extend(result)
            configurations.append(audit)
    expected_total = 6 * options.expected_episodes
    require(len(rows) == expected_total, f"Expected {expected_total} audited rendering episodes")
    file_paths = [row[extension] for row in rows for extension in FORMATS]
    require(len(set(file_paths)) == expected_total * 4,
            f"Expected {expected_total * 4} distinct four-format output files")
    report = {"schema_version": 1, "kind": "audited_fixed_seed_rendering_batch",
              "episode_count": expected_total, "episodes_per_configuration": options.expected_episodes,
              "configuration_count": 6, "output_file_count": expected_total * 4,
              "all_required_files_present": True, "all_csv_terminal_states_match": True,
              "all_success_classifications_match": True, "all_capture_events_within_radius": True,
              "is_paper_DSR_estimate": False,
              "sampling_note": f"{options.expected_episodes} predetermined rendering episodes per configuration; every outcome is retained, without success-only selection.",
              "configurations": configurations,
              "generator": {"path": repository_reference(__file__), "sha256": digest(__file__)}}
    verify_public_text(json.dumps(report, ensure_ascii=False))
    if not options.validate_only:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        verify_public_text(buffer.getvalue())
        html_content = build_html(rows, render_root)
        verify_public_text(html_content)
        atomic_write(render_root / "success_summary.csv", buffer.getvalue())
        atomic_write(render_root / "render_manifest.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
        atomic_write(render_root / "index.html", html_content)
    print(json.dumps({"validated": True, "render_root": repository_reference(render_root), "episodes": expected_total,
                      "files": expected_total * 4, "validate_only": options.validate_only,
                      "success_episodes": {f"stage{item['stage']}/{item['scale']}": item["success_episodes"]
                                           for item in configurations}}, indent=2))


if __name__ == "__main__":
    main()
