"""Fault injection by editing a copy of a stock aircraft model (Family A) and no-op rewrites (Family C)."""
from __future__ import annotations

import os
import re
import shutil

import jsbsim


def stock_aircraft_dir(aircraft: str) -> str:
    return os.path.join(jsbsim.get_default_root_dir(), "aircraft", aircraft)


def make_variant(aircraft: str, name: str, workdir: str, edit=None, src_root: str | None = None) -> str:
    """Build a mirror JSBSim root at workdir/<name>/: the stock aircraft folder is copied to aircraft/<aircraft>
    and `edit(xml_text)->xml_text` is applied to its main file; engine/, systems/, scripts/ are symlinks to the
    stock root. Returns the mirror root, to pass as root_dir=... so every relative lookup resolves as stock."""
    root = jsbsim.get_default_root_dir()
    vroot = os.path.abspath(os.path.join(workdir, name))
    ap = os.path.join(vroot, "aircraft")
    dst = os.path.join(ap, aircraft)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(ap, exist_ok=True)
    for sub in ("engine", "systems", "scripts"):
        link = os.path.join(vroot, sub)
        if not os.path.lexists(link) and os.path.isdir(os.path.join(root, sub)):
            os.symlink(os.path.join(root, sub), link)
    src = os.path.join(src_root, "aircraft", aircraft) if src_root else stock_aircraft_dir(aircraft)
    shutil.copytree(src, dst)
    main = os.path.join(dst, f"{aircraft}.xml")
    if edit is not None:
        with open(main) as f:
            txt = f.read()
        new = edit(txt)
        if new == txt:
            raise RuntimeError(f"edit for variant {name} changed nothing")
        with open(main, "w") as f:
            f.write(new)
    return vroot


def make_reverted_root(aircraft: str, name: str, workdir: str, patch_path: str) -> str:
    """Build a mirror JSBSim root at workdir/<name>/ in which a model-file fix is reversed: the aircraft folder is
    copied; engine/ and systems/ are copied when the patch touches them (symlinked otherwise); scripts/ is a symlink.
    Only the patch sections for this aircraft's folder and for the shared engine/ and systems/ files are applied,
    with `patch -R -p1`. Raises RuntimeError when a section does not reverse cleanly or nothing applies."""
    import subprocess
    import tempfile
    root = jsbsim.get_default_root_dir()
    vroot = os.path.abspath(os.path.join(workdir, name))
    if os.path.exists(vroot):
        shutil.rmtree(vroot)
    text = open(patch_path).read()
    sections = [("diff --git" + sec) for sec in text.split("diff --git")[1:]]
    keep = []
    for sec in sections:
        path = sec.split("\n", 1)[0].split(" b/")[-1].strip()
        if path.startswith(f"aircraft/{aircraft}/") or path.startswith(("engine/", "systems/")):
            keep.append((path, sec))
    if not keep:
        raise RuntimeError(f"patch {os.path.basename(patch_path)} has nothing for {aircraft}")
    os.makedirs(os.path.join(vroot, "aircraft"), exist_ok=True)
    shutil.copytree(stock_aircraft_dir(aircraft), os.path.join(vroot, "aircraft", aircraft))
    touched = {p.split("/", 1)[0] for p, _ in keep}
    for sub in ("engine", "systems", "scripts"):
        src = os.path.join(root, sub)
        if not os.path.isdir(src):
            continue
        if sub in touched:
            shutil.copytree(src, os.path.join(vroot, sub))
        else:
            os.symlink(src, os.path.join(vroot, sub))
    # sections for files the shipped data does not contain are skipped: no aircraft can load those files
    keep = [(p, sec) for p, sec in keep if os.path.exists(os.path.join(vroot, p))]
    if not keep:
        raise RuntimeError(f"patch {os.path.basename(patch_path)} touches no shipped file used by {aircraft}")
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write("".join(sec for _, sec in keep)); tmp = fh.name
    r = subprocess.run(["patch", "-R", "-p1", "-d", vroot, "-i", tmp, "--no-backup-if-mismatch", "-s"], capture_output=True, text=True)
    os.unlink(tmp)
    if r.returncode != 0:
        raise RuntimeError(f"patch {os.path.basename(patch_path)} did not reverse cleanly for {aircraft}: {r.stdout}{r.stderr}")
    return vroot


# ---- generic edit helpers -------------------------------------------------------------------

def _replace_once(txt: str, pattern: str, repl, flags=re.S) -> str:
    new, n = re.subn(pattern, repl, txt, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"pattern not found: {pattern[:60]}")
    return new


def scale_tag(tag: str, factor: float):
    """Scale the numeric content of <tag ...> value </tag> in mass_balance etc."""
    def edit(txt):
        def rep(m):
            return f"{m.group(1)}{float(m.group(2)) * factor:g}{m.group(3)}"
        return _replace_once(txt, rf"(<{tag}\b[^>]*>\s*)([-+0-9.eE]+)(\s*</{tag}>)", rep)
    return edit


def add_ixz(value: float):
    def edit(txt):
        return _replace_once(txt, r"(<izz\b[^>]*>[^<]*</izz>)", rf'\1\n        <ixz unit="SLUG*FT2"> {value:g} </ixz>')
    return edit


def scale_function_value(func_name: str, factor: float):
    """Scale the single <value> inside a coefficient function's <product>."""
    def edit(txt):
        def rep(m):
            return f"{m.group(1)}{float(m.group(2)) * factor:.6g}{m.group(3)}"
        return _replace_once(txt, rf'(<function name="{re.escape(func_name)}">.*?<value>\s*)([-+0-9.eE]+)(\s*</value>)', rep)
    return edit


def offset_table_column(func_name: str, offset: float):
    """Add `offset` to every dependent value in the (first) <tableData> of a coefficient function."""
    def edit(txt):
        m = re.search(rf'<function name="{re.escape(func_name)}">.*?<tableData>(.*?)</tableData>', txt, re.S)
        if not m:
            raise RuntimeError(f"table not found in {func_name}")
        body = m.group(1)
        rows = []
        for line in body.splitlines():
            parts = line.split()
            if len(parts) == 2:
                rows.append(f"{line[:len(line)-len(line.lstrip())]}{parts[0]}  {float(parts[1]) + offset:.4f}")
            else:
                rows.append(line)
        return txt[:m.start(1)] + "\n".join(rows) + txt[m.end(1):]
    return edit


def nonlinear_cmalpha(alpha_star_deg: float = 8.0, slope_factor_beyond: float = 0.3, cmalpha: float | None = None):
    """Replace Cmalpha = qbar*S*cbar*alpha*value by a table in alpha-deg that is linear below alpha*,
    then continues with a reduced slope (a nonlinear Cm beyond alpha-star)."""
    def edit(txt):
        m = re.search(r'<function name="aero/coefficient/Cmalpha">(.*?)</function>', txt, re.S)
        if not m:
            raise RuntimeError("Cmalpha not found")
        v = cmalpha
        if v is None:
            vm = re.search(r"<value>\s*([-+0-9.eE]+)\s*</value>", m.group(1))
            v = float(vm.group(1))
        import math
        pts = []
        for a in (-20, -10, -5, 0, 4, alpha_star_deg, alpha_star_deg + 4, alpha_star_deg + 8, alpha_star_deg + 14, 30):
            if a <= alpha_star_deg:
                cm = v * math.radians(a)
            else:
                cm = v * math.radians(alpha_star_deg) + slope_factor_beyond * v * math.radians(a - alpha_star_deg)
            pts.append(f"                            {a:6.1f}  {cm:.5f}")
        body = ("\n                <description>Pitch_moment_due_to_alpha (nonlinear beyond alpha-star, injected fault)</description>"
                "\n                <product>"
                "\n                    <property>aero/qbar-psf</property>"
                "\n                    <property>metrics/Sw-sqft</property>"
                "\n                    <property>metrics/cbarw-ft</property>"
                "\n                    <table>"
                "\n                        <independentVar>aero/alpha-deg</independentVar>"
                "\n                        <tableData>\n" + "\n".join(pts) +
                "\n                        </tableData>"
                "\n                    </table>"
                "\n                </product>\n            ")
        return txt[:m.start(1)] + body + txt[m.end(1):]
    return edit


def noop_rewrite():
    """Family C: change whitespace/comments only."""
    def edit(txt):
        return txt.replace("    ", "\t", 1) + "\n<!-- no-op rewrite -->\n"
    return edit


def chain(*edits):
    def edit(txt):
        for e in edits:
            txt = e(txt)
        return txt
    return edit


def scale_function_matching(name_regex: str, factor: float):
    """Scale the first <value> inside the first coefficient function whose name matches name_regex.
    Raises RuntimeError if no function matches (caller may skip the fault for that aircraft)."""
    def edit(txt):
        m = re.search(rf'<function name="([^"]*)">', txt)
        names = re.findall(r'<function name="([^"]*)">', txt)
        hits = [n for n in names if re.search(name_regex, n)]
        if not hits:
            raise RuntimeError(f"no function matches {name_regex}")
        return scale_function_value(hits[0], factor)(txt)
    return edit


def function_names(aircraft: str) -> list[str]:
    with open(os.path.join(stock_aircraft_dir(aircraft), f"{aircraft}.xml")) as f:
        return re.findall(r'<function name="([^"]*)">', f.read())


# ---- axis-level operators that work across naming conventions -----------------------------

def _axis_block(txt: str, axis: str):
    m = re.search(rf'<axis name="{axis}">(.*?)</axis>', txt, re.S)
    if not m:
        raise RuntimeError(f"axis {axis} not found")
    return m


def scale_axis_function(axis: str, name_regex: str, factor: float):
    """Scale the whole output of the first function in `axis` whose name matches name_regex, by inserting a
    <value>factor</value> into its top-level <product>, or wrapping its body in a product if it has none."""
    def edit(txt):
        ax = _axis_block(txt, axis)
        body = ax.group(1)
        fm = None
        for cand in re.finditer(r'<function name="([^"]*)">(.*?)</function>', body, re.S):
            if re.search(name_regex, cand.group(1)):
                fm = cand
                break
        if fm is None:
            raise RuntimeError(f"no function in {axis} matches {name_regex}")
        fbody = fm.group(2)
        pm = re.search(r"<product>", fbody)
        if pm and fbody.strip().startswith("<description>") or (pm and not fbody.strip().startswith("<")):
            new_fbody = fbody[: pm.end()] + f"\n<value>{factor:g}</value>" + fbody[pm.end():]
        elif pm and fbody.strip().startswith("<product>"):
            new_fbody = fbody[: pm.end()] + f"\n<value>{factor:g}</value>" + fbody[pm.end():]
        else:
            # description then a non-product expression (e.g. a table): wrap everything after the description
            dm = re.search(r"</description>", fbody)
            cut = dm.end() if dm else 0
            new_fbody = fbody[:cut] + f"\n<product><value>{factor:g}</value>" + fbody[cut:] + "</product>\n"
        new_body = body[: fm.start(2)] + new_fbody + body[fm.end(2):]
        return txt[: ax.start(1)] + new_body + txt[ax.end(1):]
    return edit


def add_axis_term(axis: str, name: str, coefficient: float, extra_props=("aero/qbar-psf", "metrics/Sw-sqft")):
    """Append a new function to `axis`: coefficient * product(extra_props). E.g. a lift offset of 0.1 CL."""
    def edit(txt):
        ax = _axis_block(txt, axis)
        props = "".join(f"<property>{p}</property>" for p in extra_props)
        term = f'\n<function name="{name}"><description>injected term</description><product>{props}<value>{coefficient:g}</value></product></function>\n'
        return txt[: ax.end(1)] + term + txt[ax.end(1):]
    return edit


PITCH_DAMP = r"Pitch_damp|Cmq$|Cmq_|pitch.*damp"
PITCH_STIFF = r"Pitch_alpha$|Cmalpha$|Cma$|Cma_M|Cm_alpha"
LIFT_STIFF = r"Lift_alpha$|CLalpha$|CLa$|CL_alpha|CLa_"


# ---- Family B: faults reconstructed from real JSBSim bug fixes (model-file fixes reversed on current models) ----

def revert_elevator_drag_fix():
    """Reverse JSBSim commit 8819410c (2020-12-12, 'Fix negative drag due to elevator deflection'): the drag term
    due to elevator deflection used fcs/elevator-pos-norm (signed, normalised) instead of the magnitude in radians,
    so a nose-up deflection produced negative drag. Applies to the DRAG axis only."""
    def edit(txt):
        ax = _axis_block(txt, "DRAG")
        body = ax.group(1)
        new_body, n = re.subn(r"<property>\s*fcs/mag-elevator-pos-rad\s*</property>", "<property>fcs/elevator-pos-norm</property>", body)
        if n == 0:
            raise RuntimeError("no mag-elevator-pos-rad term in DRAG axis")
        return txt[: ax.start(1)] + new_body + txt[ax.end(1):]
    return edit


def revert_ground_effect_kink_fix():
    """Reverse JSBSim commit 9c058118 (2018-03-03, 'Fix kink in change in lift due to ground effect tables for
    C172x and B737'): the table value at h/b = 0.4 was 1.055 instead of 1.028, a kink in the ground-effect factor."""
    def edit(txt):
        new, n = re.subn(r"(0\.4000?\s+)1\.0280?\b", r"\g<1>1.0550", txt, count=1)
        if n == 0:
            raise RuntimeError("ground-effect table row 0.4 / 1.028 not found")
        return new
    return edit


FAMILY_B_XML = {
    "B_elevator_drag_8819410c": revert_elevator_drag_fix,
    "B_ground_effect_9c058118": revert_ground_effect_kink_fix,
}
