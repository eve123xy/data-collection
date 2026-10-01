"""Vast.ai provisioning with spend caps, tagging and cost accounting.

    python scripts/cells/provision.py search  --cell-id <id>
    python scripts/cells/provision.py launch  --offer 123 --cells a,b,c
    python scripts/cells/provision.py destroy --instance 456
    python scripts/cells/provision.py push-creds --instance 456
    python scripts/cells/provision.py cost
    python scripts/cells/provision.py sweep

The caps live here, in code, rather than in an agent's judgement.
OPERATIONAL_LEARNINGS 4.1: burn rate, not total GPU-hours, is what kills a
campaign. 2.1 records a single hung step becoming a 2h43m billing incident.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import os
import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pins
from common import load_env, token

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plan" / "cells_v1.json"
PROVISIONS = ROOT / "plan" / "provisions.jsonl"

GPU_QUERY = {"A100": "A100_SXM4", "H100": "H100_SXM",
             "H100 (Lambda)": "H100_SXM", "H200": "H200"}


def _vast(*args):
    out = subprocess.run(["vastai", *args, "--raw"], capture_output=True,
                         text=True, timeout=180)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"_raw": out.stdout.strip(), "_err": out.stderr.strip()}


def cells():
    return json.loads(MANIFEST.read_text())


def get_cell(cell_id):
    row = next((c for c in cells() if c["cell_id"] == cell_id), None)
    if row is None:
        sys.exit(f"[FATAL] {cell_id} not in {MANIFEST}")
    return row


def ours(instances):
    """Only instances this campaign launched. Rule 8: track OUR spend alone."""
    return [i for i in instances
            if str(i.get("label") or "").startswith(pins.INSTANCE_LABEL_PREFIX)]


def live():
    d = _vast("show", "instances")
    return d if isinstance(d, list) else []


def caps_report():
    """Current usage against the hard caps. Returns (ok, lines)."""
    mine = ours(live())
    # Count everything that BILLS, not just what has finished booting. An
    # instance pulling the image reports actual_status "loading" and is charged
    # the whole time; counting only "running" let the fleet reach $40.19/hr
    # against a $35 ceiling on 2026-09-06, because three boxes were mid-pull
    # when the check ran.
    running = [i for i in mine
               if i.get("actual_status") not in ("exited", "stopped", None)]
    dph = sum(float(i.get("dph_total") or 0) for i in running)
    lines = [
        f"  instances: {len(running)} / {pins.MAX_CONCURRENT_INSTANCES}",
        f"  $/hr:      {dph:.2f} / {pins.MAX_DPH_IN_FLIGHT:.2f}",
    ]
    ok = (len(running) < pins.MAX_CONCURRENT_INSTANCES
          and dph < pins.MAX_DPH_IN_FLIGHT)
    return ok, lines, dph, len(running)


def search(cell_row, extra=""):
    """Print the exact command and its filtered results, for rule 1 approval."""
    gpu = GPU_QUERY.get(cell_row["gpu"], cell_row["gpu"])
    n = cell_row["expected_gpus"]
    # image 31.6 GB unpacked + weights + artifacts
    # cuda_max_good is what the HOST driver supports. Below pins.MIN_CUDA_MAX_GOOD
    # the campaign image cannot run CUDA at all, and both --verify-gpu and
    # bootstrap pass anyway because they only use NVML/DCGM -- so an unfiltered
    # offer burns the image pull, the weights and the vLLM start before failing.
    # gpu_ram is PER-GPU and is the only field that pins the memory variant.
    # vast's gpu_name carries no memory size for A100 ("A100 SXM4" covers both
    # 40GB and 80GB), and gpu_totalram is the AGGREGATE -- so a 2x A100 40GB box
    # reports gpu_totalram=81920, byte-identical to what a 1x A100 80GB reports
    # as gpu_ram. Filtering on the name or the aggregate cannot tell them apart,
    # and on 2026-09-04 a 2x A100 40GB was rented for cells pinned to SXM4 80GB.
    # --verify-gpu caught it before any measurement, but the rental was wasted.
    want_gb = pins.GPUS[cell_row["gpu"]]["mem_gb"]
    q = (f"gpu_name={gpu} num_gpus={n} disk_space>=100 rentable=true "
         # vast's SEARCH FILTER takes gpu_ram in GB; the field it RETURNS is in
         # MiB. Passing MiB here matches nothing and reports "0 offers", which
         # reads exactly like scarcity -- it hid every H200 on 2026-09-04.
         f"gpu_ram>={int(want_gb * 0.95)} "
         f"cuda_max_good>={pins.MIN_CUDA_MAX_GOOD} {extra}").strip()
    cmd = ["vastai", "search", "offers", q, "-o", "dph_total", "--raw"]
    print("$ " + " ".join(f"'{c}'" if " " in c else c for c in cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    try:
        offers = json.loads(out.stdout)
    except Exception:
        print(out.stdout[:400] or out.stderr[:400])
        return []
    print(f"\n{len(offers)} offer(s) for {n}x {cell_row['gpu']}:\n")
    print(f"{'id':>10} {'$/hr':>7} {'disk':>6} {'down Mbps':>10} {'rel':>6} "
          f"{'driver':>10} {'GB/gpu':>7}  location")
    usable = []
    for o in offers:
        drv = o.get("driver_version") or "0"
        try:
            ok = float(".".join(str(drv).split(".")[:2])) >= pins.MIN_DRIVER_VERSION
        except ValueError:
            ok = False
        if ok:
            usable.append(o)
    dropped = len(offers) - len(usable)
    for o in usable[:8]:
        print(f"{o['id']:>10} {o['dph_total']:>7.3f} {o.get('disk_space',0):>6.0f} "
              f"{o.get('inet_down',0):>10.0f} {o.get('reliability2',0):>6.3f} "
              f"{str(o.get('driver_version','?')):>10} "
              f"{o.get('gpu_ram',0)/1024:>7.0f}  {o.get('geolocation','?')}")
    if dropped:
        print(f"\n({dropped} offer(s) hidden: driver below "
              f"{pins.MIN_DRIVER_VERSION}, cannot run the campaign image)")
    return usable


IMAGE_GB = 20           # campaign image unpacked
WORKDIR_GB = 25         # compile cache, /data prompts, per-cell artifacts
DISK_FLOOR_GB = 100


def repo_size_gb(hf_id):
    """Actual repo size on HuggingFace, or None if it cannot be determined.

    Asked because a parameter count does NOT predict disk. meta-llama's
    Llama-3.1-70B-Instruct ships an `original/*.pth` copy alongside its
    safetensors, so the repo is 262.9 GiB where the weights are 131.4 GiB, and
    nothing downloads only half of it by default.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(hf_id, files_metadata=True, token=token())
        n = sum((f.size or 0) for f in (info.siblings or []))
        return n / 1e9 if n else None
    except Exception:
        return None


def disk_for(rows):
    """GB this grouping needs: image + every distinct model + working space.

    The preflight rule says disk is DERIVED from the cells, never inherited
    from an older plan. It used to be a flat 100 GB default, which is why a
    4x A100 TP4 group reached its Llama-70B cell with 99 GB free against 131.4
    GiB of safetensors and could not run it -- an hour of a rented box spent
    discovering a number we could have computed for free.
    """
    # cell["model"] is the plain hf_id string; pins.MODELS is keyed by
    # (family, size) so invert it once to recover size_b for the fallback.
    by_hf = {m["hf_id"]: m for m in pins.MODELS.values()}
    seen, total, unknown = set(), 0.0, []
    for r in rows:
        hf = r.get("model")
        if not isinstance(hf, str) or hf in seen:
            continue
        seen.add(hf)
        gb = repo_size_gb(hf)
        if gb is None:
            size_b = (by_hf.get(hf) or {}).get("size_b") or 0
            gb = 2.6 * size_b if size_b else 0      # bf16 + a duplicate-copy allowance
            unknown.append(hf)
        total += gb
    need = int(IMAGE_GB + WORKDIR_GB + total * 1.15) + 1      # 15% filesystem slack
    need = max(need, DISK_FLOOR_GB)
    print(f"[..] disk derived from {len(seen)} model(s): {total:.0f} GB of weights "
          f"+ {IMAGE_GB} image + {WORKDIR_GB} work -> {need} GB")
    if unknown:
        print(f"[WARN] size estimated from parameter count for: {', '.join(unknown)}")
    return need


def _offer_dph(offer_id):
    """$/hr for one offer, or None if it cannot be read."""
    try:
        out = subprocess.run(["vastai", "search", "offers", f"id={offer_id}",
                              "--raw"], capture_output=True, text=True, timeout=90).stdout
        rows = json.loads(out or "[]")
        return float(rows[0]["dph_total"]) if rows else None
    except Exception:
        return None


def launch(offer_id, cell_ids, disk=None):
    ok, lines, dph, n = caps_report()
    if not ok:
        print("[FATAL] spend cap reached, not launching:")
        print("\n".join(lines))
        sys.exit(1)

    # Check the cap against what the fleet will cost AFTER this launch, not
    # what it costs now. "current < cap" happily admits a box that takes the
    # total far past it -- which is how a $11.26/hr box was added to a fleet
    # already at $28.60 under a $35 ceiling.
    want = _offer_dph(offer_id)
    if want and dph + want > pins.MAX_DPH_IN_FLIGHT:
        print(f"[FATAL] this offer is ${want:.2f}/hr; the fleet is at "
              f"${dph:.2f}/hr and the cap is ${pins.MAX_DPH_IN_FLIGHT:.2f}/hr. "
              f"Launching would reach ${dph + want:.2f}/hr.")
        sys.exit(1)

    rows = [get_cell(c) for c in cell_ids]
    if disk is None:
        disk = disk_for(rows)
    sheets = sorted({r["sheet"] for r in rows})
    if len({(r["gpu"], r["expected_gpus"]) for r in rows}) > 1:
        sys.exit("[FATAL] cells in one provision must share GPU type and count")

    hrs = budget_hours(cell_ids)
    if hrs > pins.MAX_INSTANCE_HOURS:
        print(f"[FATAL] this grouping needs ~{hrs:.1f}h, over the "
              f"{pins.MAX_INSTANCE_HOURS}h cap. Split it into smaller "
              f"provisions - the cap is what stops a hung run becoming a "
              f"billing incident.")
        sys.exit(1)
    print(f"[..] budget ~{hrs:.1f}h of {pins.MAX_INSTANCE_HOURS}h cap")

    slug = re.sub(r"[^a-z0-9]+", "", sheets[0].lower())[:10]
    label = f"{pins.INSTANCE_LABEL_PREFIX}-{slug}-{uuid.uuid4().hex[:6]}"

    load_env()
    import os
    dhu = os.environ["DOCKERHUB_USERNAME"]
    dht = os.environ["DOCKERHUB_RO_TOKEN"]

    # Inject the ssh key AT BOOT rather than relying on `vastai attach ssh`.
    # The account is a TEAM account, so `vastai create ssh-key` is refused
    # outright and keys go per-instance -- but attach is unreliable: on
    # 2026-09-02/03 three separate boxes reported the key "already associated"
    # and still refused it, each costing a destroy-and-relaunch cycle while
    # billing. onstart injection has worked every time it was used. The key
    # must be PASSPHRASE-LESS; BatchMode cannot unlock an encrypted one, and
    # the failure then reads like a server rejection.
    import os as _os
    pub = _os.environ.get("CAMPAIGN_SSH_PUBKEY", "")
    if not pub:
        key = _os.environ.get("CAMPAIGN_SSH_KEY", "")
        if key and Path(key + ".pub").exists():
            pub = Path(key + ".pub").read_text().strip()
    onstart = []
    if pub:
        onstart = ["--onstart-cmd",
                   "mkdir -p /root/.ssh && echo '" + pub +
                   "' >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh "
                   "&& chmod 600 /root/.ssh/authorized_keys"]
    else:
        print("[WARN] no CAMPAIGN_SSH_PUBKEY/CAMPAIGN_SSH_KEY - falling back to "
              "`vastai attach ssh`, which has failed on 3 of the last 8 boxes")

    res = _vast("create", "instance", str(offer_id),
                "--image", pins.CAMPAIGN_IMAGE,
                "--login", f"-u {dhu} -p {dht} docker.io",
                "--disk", str(disk), "--label", label,
                *onstart,
                "--ssh", "--direct", "--cancel-unavail")
    iid = res.get("new_contract")
    if not iid:
        print("[FATAL] launch failed:", json.dumps(res)[:300])
        sys.exit(1)

    rec = {"instance": iid, "label": label, "offer": offer_id,
           "cells": cell_ids, "sheets": sheets, "disk": disk,
           "launched_ts": time.time()}
    with open(PROVISIONS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[OK] instance {iid} label {label} for {len(cell_ids)} cell(s)")
    print("     NOTE: the launch response contains an instance_api_key - never echo it")
    return rec


def destroy(instance_id):
    r = _vast("destroy", "instance", str(instance_id), "-y")
    still = any(i.get("id") == int(instance_id) for i in live())
    print(f"[{'FAIL' if still else 'OK'}] destroy {instance_id}"
          + ("  STILL PRESENT" if still else ""))
    return not still


def _key_path():
    """The campaign ssh private key, from .env or the conventional location."""
    load_env()
    k = os.environ.get("CAMPAIGN_SSH_KEY", "")
    return k or str(Path.home() / ".ssh" / "campaign_ed25519")


def _ssh_targets(instance_id):
    """Every route to the container, best-first: [(user, host, port), ...].

    vast exposes TWO independent paths -- a proxy (sshN.vast.ai:PORT) and the
    machine's own direct port mapping -- and they fail INDEPENDENTLY. On
    2026-09-04 the proxy refused for ~20 minutes while the container was
    healthy and the direct route worked the whole time; earlier the same day
    several boxes were abandoned as unreachable after testing only the one
    route `vastai ssh-url` happened to return. Trying both turns a coin-flip
    into a retry.
    """
    out = []
    url = subprocess.run(["vastai", "ssh-url", str(instance_id)],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    if url.startswith("ssh://"):
        user_host, _, port = url[len("ssh://"):].partition(":")
        user, _, host = user_host.partition("@")
        out.append((user or "root", host or user_host, port or "22"))

    for i in live():
        if i.get("id") != int(instance_id):
            continue
        host, sp = i.get("ssh_host"), i.get("ssh_port")
        if host and sp:
            t = ("root", host, str(sp))
            if t not in out:
                out.append(t)
        direct = ((i.get("ports") or {}).get("22/tcp") or [])
        ip = i.get("public_ipaddr")
        for d in direct:
            hp = d.get("HostPort")
            if ip and hp:
                t = ("root", ip.strip(), str(hp))
                if t not in out:
                    out.append(t)
    return out


def _ssh_target(instance_id):
    """First reachable route, or None. Kept for callers expecting one target."""
    ts = _ssh_targets(instance_id)
    return ts[0] if ts else None


def push_creds(instance_id):
    """Deliver secrets and the current scripts/ tree to a live instance.

    The image deliberately carries neither. On 2026-09-02 a `COPY scripts/`
    with no .dockerignore baked the Google service-account private key and
    scripts/.env into a published layer; secrets now travel over ssh at run
    time instead. The same copy refreshes scripts/, so a script edit never
    needs a 9.6 GB rebuild -- which is the other half of that same incident,
    where the published image predated the stage-folder reorg.
    """
    targets = _ssh_targets(instance_id)
    if not targets:
        print(f"[FAIL] no ssh endpoint for {instance_id}")
        return False
    # Probe every route before shipping anything. The proxy and the direct port
    # mapping fail independently, so "unreachable" is only true when BOTH are.
    user = hostname = port = None
    for u, h, pt in targets:
        probe = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=20", "-i", _key_path(), "-p", str(pt),
             f"{u}@{h}", "true"], capture_output=True, text=True)
        state = "OK" if probe.returncode == 0 else (
            probe.stderr.strip().splitlines()[-1][:60] if probe.stderr else "failed")
        print(f"     route {h}:{pt} -> {state}")
        if probe.returncode == 0:
            user, hostname, port = u, h, str(pt)
            break
    if hostname is None:
        print(f"[FAIL] {instance_id}: no route accepted the key "
              f"({len(targets)} tried)")
        return False

    env = ROOT / "scripts" / ".env"
    if not env.exists():
        print("[FAIL] scripts/.env not found; nothing to deliver")
        return False

    # Resolve the credentials file the way the scripts themselves do.
    load_env()
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    cred = Path(cred) if cred else None
    if cred and not cred.is_absolute():
        cred = ROOT / "scripts" / cred.name
    if not cred or not cred.exists():
        print("[FAIL] GOOGLE_APPLICATION_CREDENTIALS does not resolve to a file")
        return False

    ssh = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "BatchMode=yes", "-p", port]
    # The vast account is a TEAM account: `vastai create ssh-key` is refused
    # outright, so keys are attached per instance with `vastai attach ssh`.
    # That works, but ONLY with a passphrase-less key -- BatchMode cannot
    # unlock an encrypted one, and the failure reads like a server rejection
    # (see OPERATIONAL_LEARNINGS 2.14). Point CAMPAIGN_SSH_KEY at that key.
    key = os.environ.get("CAMPAIGN_SSH_KEY", "")
    if key:
        ssh = ["-i", key, "-o", "IdentitiesOnly=yes"] + ssh
    dest = f"{user}@{hostname}:/opt/campaign/scripts/"

    # scripts/ first, then secrets. Deliberately tar-over-ssh rather than
    # rsync: macOS ships openrsync (protocol 29, "2.6.9 compatible"), whose
    # exclude rules the remote rsync 3.2.7 cannot parse -- it dies with
    # "[Receiver] buffer overflow: recv_rules (exclude.c)". tar has no such
    # version coupling and needs nothing installed on either side.
    # The tree is wiped first so the instance matches local exactly; the image
    # ships /opt/campaign/scripts empty, so there is nothing else to lose.
    # COPYFILE_DISABLE=1 is load-bearing on macOS. bsdtar SYNTHESISES `._*`
    # AppleDouble members to carry extended attributes, so they do not exist on
    # disk and `--exclude ._*` cannot match them -- the instance ends up with 30
    # real scripts plus 30 `._` shadows, and every `find | wc -l` gate then
    # reports 60. That made the delivery check read as a stale tree on several
    # boxes and trained agents to wave it through, which is exactly the signal
    # the check exists to give.
    tar_env = {**os.environ, "COPYFILE_DISABLE": "1"}
    tar = subprocess.Popen(
        ["tar", "czf", "-", "-C", str(ROOT / "scripts"),
         "--exclude", ".env", "--exclude", "*.json", "--exclude", "__pycache__",
         "--exclude", "._*", "--exclude", ".DS_Store",
         "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=tar_env)
    rc = subprocess.run(
        ["ssh", *ssh, f"{user}@{hostname}",
         "rm -rf /opt/campaign/scripts && mkdir -p /opt/campaign/scripts && "
         "tar xzf - -C /opt/campaign/scripts"],
        stdin=tar.stdout, capture_output=True, text=True, timeout=300)
    tar.stdout.close()
    if tar.wait() != 0 or rc.returncode != 0:
        err = (rc.stderr or "") + (tar.stderr.read().decode() if tar.stderr else "")
        # Strip the vast MOTD, which otherwise fills the whole error budget.
        err = "\n".join(l for l in err.splitlines()
                         if l.strip() and "vast.ai" not in l
                         and "Have fun" not in l and "Permanently added" not in l)
        print(f"[FAIL] copy scripts: {err.strip()[:300]}")
        return False

    # scp spells the port -P; -p means "preserve times". Reusing the ssh option
    # list verbatim made scp read the port number as a local filename.
    scp_opts = ["-P" if o == "-p" else o for o in ssh]
    # .env carries GOOGLE_APPLICATION_CREDENTIALS as a LOCAL absolute path, which
    # cannot resolve on the instance -- drive.py hands it straight to
    # from_service_account_file, so the first upload of the run dies with
    # FileNotFoundError AFTER the GPU time is already spent. Rewrite that one
    # line to the on-instance path as we deliver it. Local .env is untouched.
    # Staged OUTSIDE the repo: this file holds the same secrets as .env, and a
    # crash between write and unlink would leave it in a tree that `git add -A`
    # sweeps. .gitignore now covers it as a second line of defence, but the
    # first line is not putting it there at all.
    staged = Path(tempfile.mkdtemp(prefix="llmpl-creds-")) / ".env"
    lines = []
    for line in env.read_text().splitlines():
        if line.strip().startswith("GOOGLE_APPLICATION_CREDENTIALS"):
            line = f"GOOGLE_APPLICATION_CREDENTIALS=/opt/campaign/scripts/{cred.name}"
        lines.append(line)
    staged.write_text("\n".join(lines) + "\n")

    try:
        for f, remote_name in ((staged, ".env"), (cred, cred.name)):
            rc = subprocess.run(
                ["scp", *scp_opts, str(f), dest.rstrip("/") + "/" + remote_name],
                capture_output=True, text=True, timeout=120)
            if rc.returncode != 0:
                print(f"[FAIL] scp {remote_name}: {rc.stderr.strip()[:200]}")
                return False
    finally:
        staged.unlink(missing_ok=True)
        try:
            staged.parent.rmdir()
        except OSError:
            pass

    # Stamp the commit the delivered tree came from. The instance has no .git,
    # so without this record_run.py records git_commit=null and per-run code
    # provenance is silently lost -- which is what the smoke/pin workflow is
    # built on.
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts",
                                "plan/cells_v1.json"], cwd=str(ROOT),
                               capture_output=True, text=True,
                               timeout=30).stdout.strip()
    except Exception:
        commit, dirty = "", "git unavailable"
    if not commit:
        print("[FAIL] cannot resolve local git HEAD; refusing to deliver "
              "an un-stamped tree")
        return False
    if dirty:
        # We deliver the WORKING TREE but stamp HEAD. If they differ, every run
        # record from this instance claims a provenance that does not describe
        # the code that produced it -- silently, and in exactly the cells meant
        # to pin an instrument. Refuse rather than lie.
        print("[FAIL] working tree is dirty; CAMPAIGN_COMMIT would not describe "
              "what is delivered. Commit or stash first:")
        for line in dirty.splitlines()[:10]:
            print("       " + line)
        return False
    rc = subprocess.run(
        ["ssh", *ssh, f"{user}@{hostname}",
         f"printf '%s' {commit} > /opt/campaign/scripts/CAMPAIGN_COMMIT"],
        capture_output=True, text=True, timeout=60)
    if rc.returncode != 0:
        print(f"[FAIL] could not stamp CAMPAIGN_COMMIT: {rc.stderr.strip()[:150]}")
        return False

    # Data files the image does not carry. cell.py --verify-gpu dies on a missing
    # manifest, which is another failure that only bites after money is spent.
    rc = subprocess.run(
        ["ssh", *ssh, f"{user}@{hostname}", "mkdir -p /opt/campaign/plan"],
        capture_output=True, text=True, timeout=60)
    for data in (ROOT / "plan" / "cells_v1.json",):
        if not data.exists():
            print(f"[FAIL] {data} missing locally"); return False
        rc = subprocess.run(
            ["scp", *scp_opts, str(data),
             f"{user}@{hostname}:/opt/campaign/plan/{data.name}"],
            capture_output=True, text=True, timeout=120)
        if rc.returncode != 0:
            print(f"[FAIL] scp {data.name}: {rc.stderr.strip()[:200]}")
            return False

    # Verify without ever printing a value.
    chk = subprocess.run(
        ["ssh", *ssh, f"{user}@{hostname}",
         "cd /opt/campaign/scripts && "
         "echo scripts=$(ls */*.py 2>/dev/null | wc -l) && "
         "echo env=$([ -s .env ] && echo present || echo MISSING) && "
         "echo cred=$([ -s " + cred.name + " ] && echo present || echo MISSING) && "
         "echo commit=$(cat CAMPAIGN_COMMIT 2>/dev/null | cut -c1-12 || echo MISSING) && "
         "echo manifest=$([ -s /opt/campaign/plan/cells_v1.json ] && echo present || echo MISSING) && "
         "chmod 600 .env " + cred.name],
        capture_output=True, text=True, timeout=120)
    print("[OK] push-creds " + instance_id)
    for line in chk.stdout.strip().splitlines():
        print("     " + line.strip())
    return "MISSING" not in chk.stdout


def cost():
    """Spend for OUR instances only (rule 8), from the local provision log."""
    recs = ([json.loads(l) for l in PROVISIONS.read_text().splitlines() if l.strip()]
            if PROVISIONS.exists() else [])
    by_id = {i.get("id"): i for i in live()}
    total_live = 0.0
    print(f"{'instance':>10} {'label':<24} {'status':>10} {'$/hr':>7} {'hrs':>6} {'cells':>6}")
    for r in recs:
        i = by_id.get(r["instance"])
        # actual_status is absent for the first seconds of a launch; None here
        # crashed the whole cost report on a format spec.
        status = (i.get("actual_status") or "starting") if i else "gone"
        dph = float(i.get("dph_total") or 0) if i else 0.0
        hrs = (time.time() - r["launched_ts"]) / 3600
        if status == "running":
            total_live += dph
        print(f"{r['instance']:>10} {r['label']:<24} {status:>10} {dph:>7.3f} "
              f"{hrs:>6.2f} {len(r['cells']):>6}")
    print(f"\nlive burn rate: ${total_live:.2f}/hr")
    _, lines, _, _ = caps_report()
    print("caps:"); print("\n".join(lines))
    return total_live


def budget_hours(cell_ids):
    """Wall-clock a grouping needs: warm-up + window + drain per cell, plus a
    fixed allowance for image pull, model load and vLLM startup."""
    by_id = {c["cell_id"]: c for c in cells()}
    total = 20 * 60
    for cid in cell_ids:
        c = by_id.get(cid)
        if c:
            total += pins.WARMUP_S + c["workload"]["duration"] + 120
    return total / 3600


def sweep(destroy_them=True):
    """Instances WE launched that are past the wall-clock cap.

    The vast account is shared with the rest of the team, so anything without
    our label is somebody else's and is left strictly alone - not destroyed,
    not flagged.
    """
    recs = ({json.loads(l)["instance"]: json.loads(l)
             for l in PROVISIONS.read_text().splitlines() if l.strip()}
            if PROVISIONS.exists() else {})
    problems, skipped = [], 0
    for i in live():
        label = str(i.get("label") or "")
        if not label.startswith(pins.INSTANCE_LABEL_PREFIX):
            skipped += 1        # someone else on the team account
            continue
        r = recs.get(i["id"])
        if r:
            hrs = (time.time() - r["launched_ts"]) / 3600
            if hrs > pins.MAX_INSTANCE_HOURS:
                problems.append((i["id"], f"{hrs:.1f}h exceeds the "
                                          f"{pins.MAX_INSTANCE_HOURS}h cap"))
    if skipped:
        print(f"  ({skipped} instance(s) not ours - team account, left alone)")
    if not problems:
        print("[OK] nothing of ours needs sweeping")
    for iid, why in problems:
        print(f"[SWEEP] {iid}: {why}")
        if destroy_them:
            # The 3h cap is documented as "auto-destroy past this, whatever it
            # is doing" but sweep only ever REPORTED, so the cap was enforced by
            # the orchestrator remembering to look. On 2026-09-03 an idle H200
            # ran 4.52h against the cap while waiting on a reply -- about $7 for
            # nothing. A limit that depends on someone remembering is not a
            # limit.
            print(f"         destroying {iid} (past the {pins.MAX_INSTANCE_HOURS}h cap)")
            destroy(iid)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["search", "launch", "destroy",
                                       "push-creds", "cost", "sweep"])
    ap.add_argument("--cell-id"); ap.add_argument("--cells")
    ap.add_argument("--offer"); ap.add_argument("--instance")
    ap.add_argument("--disk", type=int, default=None,
                    help="override; by default derived from the cells' models")
    ap.add_argument("--dry-run", action="store_true",
                    help="sweep: report over-cap instances without destroying them")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    if a.action == "search":
        search(get_cell(a.cell_id), a.extra)
    elif a.action == "launch":
        launch(a.offer, [c for c in (a.cells or "").split(",") if c], a.disk)
    elif a.action == "destroy":
        sys.exit(0 if destroy(a.instance) else 1)
    elif a.action == "push-creds":
        sys.exit(0 if push_creds(a.instance) else 1)
    elif a.action == "cost":
        cost()
    elif a.action == "sweep":
        sweep(destroy_them=not a.dry_run)


if __name__ == "__main__":
    main()
