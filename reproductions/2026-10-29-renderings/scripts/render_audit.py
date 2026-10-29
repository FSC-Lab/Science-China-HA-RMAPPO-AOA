"""Shared passive auditing and exports for the native Stage I/II renderers."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CSV_FIELDS = ["episode", "step", "category", "entity_type", "entity_index", "x", "y",
              "ellipse_width", "ellipse_height", "ellipse_angle_deg", "time", "captured",
              "frame_phase", "sticky_breach"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def add_render_audit_arguments(parser, stage):
    parser.add_argument("--seed", type=int, default=61000,
                        help="Episode k independently resets Python/NumPy/PyTorch to seed+k-1.")
    parser.add_argument("--legend_loc", choices=["paper", "best", "upper right", "upper left", "lower left", "lower right"],
                        default="paper", help="paper follows Fig7/8/9 legend corners without changing legend items.")
    if stage == 1:
        parser.add_argument("--csv", action="store_true", help="Export native frames and an audited terminal state to CSV.")


def legend_location(args, stage):
    location = getattr(args, "legend_loc", "paper")
    if location != "paper":
        return location
    return "upper right" if stage == 1 else ("upper left" if args.defenders == 5 and args.intruders == 2 else "lower left")


def strict_load_policy(policy, actor_paths, critic_paths):
    """Require complete architecture matches; record and verify every tensor."""
    report = {}
    for role, module, paths in (("actor", policy.actor, actor_paths), ("critic", policy.critic, critic_paths)):
        path = next((Path(candidate) for candidate in paths if Path(candidate).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No complete {role} checkpoint among {paths}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        current = module.state_dict()
        if set(state) != set(current):
            raise ValueError(f"{path}: checkpoint keys mismatch; missing={set(current)-set(state)}, extra={set(state)-set(current)}")
        mismatched = [key for key in current if current[key].shape != state[key].shape]
        if mismatched:
            raise ValueError(f"{path}: checkpoint shape mismatch: {mismatched}")
        module.load_state_dict(state, strict=True)
        if not all(torch.equal(value.cpu(), state[key].cpu()) for key, value in module.state_dict().items()):
            raise ValueError(f"{path}: loaded tensor values differ from checkpoint")
        module.eval()
        origin_blob = None
        try:
            relative = path.resolve().relative_to(ROOT)
            origin_blob = subprocess.check_output(["git", "rev-parse", f"origin/main:{relative}"], cwd=ROOT,
                                                   stderr=subprocess.DEVNULL, text=True).strip()
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
        local_blob = subprocess.check_output(["git", "hash-object", str(path.resolve())], cwd=ROOT, text=True).strip()
        report[role] = {"path": str(path.resolve()), "sha256": sha256(path), "tensor_count": len(state),
                        "strict": True, "all_keys_match": True, "all_shapes_match": True, "all_tensor_values_equal": True,
                        "origin_main_blob": origin_blob, "local_blob": local_blob,
                        "matches_origin_main": local_blob == origin_blob if origin_blob else None}
    policy._render_checkpoint_report = report


def runtime_world_parameters(env, cfg):
    mapping = {"world_r": "_pz_world_r", "protected_r": "_pz_r", "capture_r": "_pz_capture",
               "capture_reward": "_pz_capture_reward", "entry_penalty": "_pz_entry_penalty",
               "bearing_sigma0": "_aoa_sigma0", "bearing_r0": "_aoa_r0", "cpf_num_particles": "_cpf_num_particles",
               "cpf_sigma_a": "_cpf_sigma_a", "threat_lambda": "_threat_lambda", "threat_tau_max": "_threat_tau_max",
               "intruder_sense_radius": "_intruder_sense_radius", "intruder_max_neighbors": "_intruder_max_neighbors"}
    actual = {"dt": float(env.world.dt), "episode_length": int(env.world.world_length),
              "protected_center": env.world._pz_center.tolist(),
              "agent_max_speeds": [float(agent.max_speed) for agent in env.world.agents]}
    if abs(actual["dt"]-.1) > 1e-10 or actual["episode_length"] != cfg.episode_length:
        raise ValueError("Actual native world dt/horizon differs from rendering configuration")
    for public, private in mapping.items():
        if hasattr(env.world, private):
            actual[public] = getattr(env.world, private)
            if hasattr(cfg, public) and abs(float(actual[public])-float(getattr(cfg, public))) > 1e-10:
                raise ValueError(f"Actual world {public} differs from cfg")
    expected_speeds = [cfg.defender_max_speed]*cfg.num_defenders + [cfg.intruder_max_speed]*cfg.num_intruders
    if any(abs(a-b) > 1e-10 for a, b in zip(actual["agent_max_speeds"], expected_speeds)):
        raise ValueError("Actual agent speed caps differ from cfg")
    extras = ("damping", "_pz_time_penalty", "_pz_vel_penalty", "_pz_distance_reward", "_pz_formation_reward",
              "_pz_threat_weight_reward", "_pz_intercept_reward", "_cpf_init_pos_std", "_cpf_init_vel_std",
              "_cpf_init_acc_std", "_cpf_ess_ratio", "_threat_tau_step")
    actual["retained_repository_extras"] = {name: getattr(env.world, name) for name in extras if hasattr(env.world, name)}
    actual["config_matches_actual_world"] = True
    return actual


class EpisodeAudit:
    """Observe env.step without altering its arguments, return values or timing."""
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.original_step = env.step
        self.states, self.capture_events = [], []

    def _snapshot(self, step):
        world = self.env.world
        positions = np.asarray([agent.state.p_pos for agent in world.agents], dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite physical positions")
        return {"step": step, "time": float(step*world.dt), "positions": positions.tolist(),
                "captured_ids": sorted(int(i) for i in getattr(world, "_pz_already_captured", set())),
                "sticky_breach": bool(getattr(world, "_pz_zone_entered", False) or getattr(world, "_pz_zone_entry", False))}

    def __enter__(self):
        self.env.step = self._step
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.step = self.original_step

    def _step(self, actions):
        if not self.states:
            self.states.append(self._snapshot(0))
        before = set(self.states[-1]["captured_ids"])
        result = self.original_step(actions)
        state = self._snapshot(len(self.states))
        info = result[3][0] if result[3] else {}
        state["returned_defense_success"] = bool(info.get("defense_success", False))
        state["returned_attack_success"] = bool(info.get("attack_success", False))
        state["returned_done_all"] = bool(np.asarray(result[2]).all())
        positions = np.asarray(state["positions"])
        for index in sorted(set(state["captured_ids"])-before):
            intruder = positions[self.cfg.num_defenders+index]
            distances = np.linalg.norm(positions[:self.cfg.num_defenders]-intruder, axis=-1)
            nearest = int(np.argmin(distances))
            zone_distance = float(np.linalg.norm(intruder-self.env.world._pz_center))
            self.capture_events.append({"intruder_index": index, "step": state["step"], "time": state["time"],
                "defender_index": nearest, "distance": float(distances[nearest]),
                "intruder_position": intruder.tolist(), "defender_position": positions[nearest].tolist(),
                "zone_center_distance": zone_distance, "zone_clearance": zone_distance-self.cfg.protected_r})
        self.states.append(state)
        return result

    def summarize(self, data, episode, seed):
        m, n = self.cfg.num_defenders, self.cfg.num_intruders
        if len(self.states) != len(data["frames"])+1:
            raise ValueError("Native frame count and passive step observer disagree")
        for index, frame in enumerate(data["frames"]):
            recorded = np.concatenate([frame["def"], frame["intr"]], axis=0)
            if not np.array_equal(recorded, np.asarray(self.states[index]["positions"])):
                raise ValueError("Native plotted positions and passive physical state disagree")
        positions = np.asarray([state["positions"] for state in self.states])
        intruders = positions[:, m:]
        center = self.env.world._pz_center
        sampled_distances = np.linalg.norm(intruders-center, axis=-1)
        starts, ends = intruders[:-1], intruders[1:]
        direction = ends-starts
        projection = np.sum((center-starts)*direction, axis=-1)/np.maximum(np.sum(direction*direction, axis=-1), 1e-30)
        closest = starts+np.clip(projection, 0, 1)[..., None]*direction
        segment_distances = np.linalg.norm(closest-center, axis=-1)
        sticky = any(state["sticky_breach"] for state in self.states)
        sampled_breach = bool((sampled_distances <= self.cfg.protected_r).any())
        segment_breach = bool((segment_distances <= self.cfg.protected_r).any())
        attack = sticky or sampled_breach or segment_breach
        captured = self.states[-1]["captured_ids"]
        all_captured = captured == list(range(n))
        returned_success = data["stats"]["defense_success"] > .5
        capture_distances_valid = (len(self.capture_events) == n and
            all(event["distance"] <= self.cfg.capture_r+1e-7 for event in self.capture_events))
        success = returned_success and all_captured and capture_distances_valid and not attack
        terminal_step = self.states[-1]["step"]
        return {"episode": episode, "seed": seed, "verified_defense_success": bool(success),
                "all_captured": all_captured, "captured_ids": captured, "attack_success": bool(attack),
                "timeout": bool(not success and not attack and terminal_step >= self.cfg.episode_length),
                "capture_events": self.capture_events, "capture_distances_valid": capture_distances_valid,
                "native_stats": data["stats"], "returned_defense_success": returned_success,
                "returned_attack_success": data["stats"]["attack_success"] > .5,
                "terminal_step": terminal_step, "plot_frame_count": len(data["frames"]),
                "final_positions": self.states[-1]["positions"],
                "final_nearest_defender_distances_by_intruder": np.linalg.norm(positions[-1, :m, None]-positions[-1, None, m:], axis=-1).min(axis=0).tolist(),
                "min_sampled_clearance_by_intruder": (sampled_distances.min(axis=0)-self.cfg.protected_r).tolist(),
                "min_segment_clearance_by_intruder": (segment_distances.min(axis=0)-self.cfg.protected_r).tolist(),
                "cumulative_sticky_breach": sticky, "sampled_position_breach": sampled_breach,
                "interpolated_segment_breach": segment_breach,
                "native_frames_match_audit_states": True,
                "terminal_state_note": "Native plot frames precede step; separately audited terminal world is exported only as terminal_agent CSV rows"}


def export_episode_csv(data, args, output_root, episode_idx, stage):
    path = Path(output_root)/f"{args.algo}_{args.defenders}v{args.intruders}_stage{stage}_ep{episode_idx+1:02d}.csv"
    audit = data.get("_render_audit")
    states = audit.states if audit is not None else []
    dt = float(audit.env.world.dt) if audit is not None else .1
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for step, frame in enumerate(data.get("frames", [])):
            state = states[step] if states else {"captured_ids": [], "sticky_breach": False}
            for entity_type, positions in (("defender", frame.get("def", [])), ("intruder", frame.get("intr", []))):
                for index, position in enumerate(positions):
                    captured = int(entity_type == "intruder" and index in state["captured_ids"])
                    writer.writerow([episode_idx+1, step, "agent", entity_type, index, float(position[0]), float(position[1]),
                                     "", "", "", step*dt, captured, "native_pre_step", int(state["sticky_breach"])])
            for index, (center, width, height, angle) in enumerate(frame.get("ellipses", [])):
                writer.writerow([episode_idx+1, step, "ellipse", "intruder", index, float(center[0]), float(center[1]),
                                 float(width), float(height), float(angle), step*dt,
                                 int(index in state["captured_ids"]), "native_pre_step", int(state["sticky_breach"])])
        if states:
            state = states[-1]
            for index, position in enumerate(state["positions"]):
                is_intruder = index >= args.defenders
                entity_index = index-args.defenders if is_intruder else index
                writer.writerow([episode_idx+1, state["step"], "terminal_agent", "intruder" if is_intruder else "defender",
                                 entity_index, float(position[0]), float(position[1]), "", "", "", state["time"],
                                 int(is_intruder and entity_index in state["captured_ids"]), "terminal_post_step", int(state["sticky_breach"])])
    return path


def archive_case_outputs(output, prefix):
    existing = sorted(path for path in output.glob(prefix+"_ep*") if path.is_file() and path.suffix.lower() in (".png", ".gif", ".csv", ".pdf"))
    summary = output/(prefix+"_summary.json")
    if summary.is_file():
        existing.append(summary)
    if not existing:
        return {"directory": None, "files": []}
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+prefix
    directory = output/"_archive"/name
    directory.mkdir(parents=True)
    files = []
    for path in existing:
        files.append({"original": str(path), "archived": str(directory/path.name), "sha256": sha256(path)})
        shutil.move(str(path), str(directory/path.name))
    write_json(directory/"archive_manifest.json", {"case": prefix, "files": files})
    return {"directory": str(directory), "files": files}


def source_record(stage):
    package_scripts = Path(__file__).resolve().parent.relative_to(ROOT)
    names = [str(package_scripts / f"render_protected_zone_stage{stage}.py"), str(package_scripts / "render_audit.py"), "config.py",
             "envs/mpe/MPE_env.py", "envs/mpe/environment.py", "envs/mpe/core.py",
             f"envs/mpe/scenarios/protected_zone_stage{stage}.py", "boa/cpf.py", "boa/threat.py",
             "algorithms/r_mappo/algorithm/rMAPPOPolicy.py", "algorithms/r_mappo/algorithm/r_actor_critic.py",
             "algorithms/utils/act.py", "algorithms/utils/distributions.py", "algorithms/utils/mlp.py", "algorithms/utils/rnn.py"]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: sha256(ROOT/name) for name in names}}


def run_rendering(stage, args, configure_env, resolve_model_dir, build_policies,
                  collect_episode, draw_static_episode, render_gif, export_csv, plt):
    if args.episodes < 1 or args.episode_length < 1:
        raise ValueError("Episodes and episode_length must be positive")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(stage=stage, args=args).resolve()
    env, cfg = configure_env(args)
    try:
        defender, intruder = build_policies(env, cfg, args, model_dir)
        prefix = f"{args.algo}_{args.defenders}v{args.intruders}_stage{stage}"
        checkpoint_hashes = {f"{family}_{role}": value
            for family, policy in (("defender", defender), ("intruder", intruder))
            for role, value in policy._render_checkpoint_report.items()}
        worldparams = runtime_world_parameters(env, cfg)
        backup = archive_case_outputs(output, prefix)
        metadata = {
            "kind": "native_renderer_independent_seeded_episodes", "stage": stage,
            "created_utc": datetime.now(timezone.utc).isoformat(), "config": vars(cfg), "worldparams": worldparams,
            "checkpoint_hashes": checkpoint_hashes, "model_dir": str(model_dir), "base_seed": args.seed,
            "episode_seed_rule": "base_seed + episode - 1; reset Python/NumPy/PyTorch independently",
            "requested_episodes": args.episodes, "renderer_args": vars(args), "source": source_record(stage),
            "runtime": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "executable": sys.executable,
                        "python": sys.version, "platform": platform.platform(),
                        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "matplotlib")}},
            "legend_location": legend_location(args, stage), "archived_previous_outputs": backup,
            "native_rollout_semantics": "StageI ignores discrete assignments; StageII uses original greedy rebalance; neither native collect uses training-runner action masks",
            "notes": [
                "All requested episodes are retained, including failures; no success-only screening.",
                "Actual world parameters are checked against cfg; no training occurs during rendering.",
                "Original native draw_static_episode and render_gif geometry/markers/styles are retained; legend location is configurable.",
                "Native end markers represent the common last plotted frame, not individual capture-time positions of defenders.",
                "Capture success requires returned defense success, every intruder captured within capture_r, and no sampled, segment or sticky breach, including terminal world state.",
                "StageII radius1/K2 sensing is the original renderer default, not a parameter specified by paper tables; paper Eq34 describes global observations.",
                "Released truth-assisted CPF initialization, MPE force integration, callback timing and reward differences are retained; no claim that all paper formulas are identical.",
                "StageII native renderer does not execute training runner dense reward shaping; recorded tau_switch and dense weights do not imply online runner execution.",
                "Archived checkpoint original training seed and complete training configuration are unknown.",
            ],
        }
        summary = {"metadata": metadata, "episodes": [], "success_episodes": []}
        summary_path = output/(prefix+"_summary.json")
        write_json(summary_path, summary)
        for index in range(args.episodes):
            seed = args.seed+index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            with EpisodeAudit(env, cfg) as audit, torch.inference_mode():
                data = collect_episode(env, defender, intruder, args, cfg)
            row = audit.summarize(data, index+1, seed)
            data["_render_audit"] = audit
            base = output/f"{prefix}_ep{index+1:02d}"
            row["files"] = {"png": str(base.with_suffix(".png")), "gif": None, "csv": None, "pdf": None}
            plt.rcdefaults()
            fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
            draw_static_episode(ax, data, cfg, args, index)
            fig.tight_layout()
            fig.savefig(row["files"]["png"], dpi=args.dpi, bbox_inches="tight")
            if args.pdf:
                row["files"]["pdf"] = str(base.with_suffix(".pdf"))
                fig.savefig(row["files"]["pdf"], bbox_inches="tight")
            plt.close(fig)
            if args.gif:
                row["files"]["gif"] = str(base.with_suffix(".gif"))
                render_gif(data["frames"], cfg, args, Path(row["files"]["gif"]))
            if args.csv:
                row["files"]["csv"] = str(export_csv(data, args, output, index).resolve())
            row["file_sha256"] = {kind: sha256(path) for kind, path in row["files"].items() if path is not None}
            summary["episodes"].append(row)
            summary["success_episodes"] = [episode["episode"] for episode in summary["episodes"] if episode["verified_defense_success"]]
            write_json(summary_path, summary)
            print(f"{prefix} ep{index+1:02d} seed={seed} verified_success={row['verified_defense_success']} "
                  f"captured={len(row['captured_ids'])}/{args.intruders}; summary={summary_path}", flush=True)
        return summary
    finally:
        env.close()
