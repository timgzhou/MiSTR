"""A run's configuration: plain dataclasses composed into one RunConfig, saved as JSON next to
the checkpoint. RunConfig.check raises on a composition that cannot run, before training.

    cfg = RunConfig(task="pastis", model=ModelConfig(coarse_patch=16, window_px=8),
                    select=SelectConfig(percentage=0.25),
                    train=TrainConfig(epochs=100, warmup="oracle", warmup_epochs=50,
                                      warmup_windows=8))
    cfg.name()   # mistr_pastis_S16Tsea+S8Tmon_q25_woracle50_seed0_<hash>

Three kinds of run: MiSTR (the default), a fixed-resolution model (fixed=True: coarse tokens
only, S<p>T<sea|mon>), a published baseline (baseline=BaselineConfig(...)).

Run names: a readable tag + a hash of every field set away from its default (not the
NOT_HASHED bookkeeping fields), so adding a field with a default never renames a run. The
paper's runs: exp.mistr.paper_configs (trained by exp.mistr.paper_runs).
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

ATTENTION = ("one_way", "fine_reads_coarse", "joint")
# the encoder's names for them (mistr.core.encoder.OneWayEncoder attn)
ENCODER_ATTN = {"one_way": "cross_hr", "fine_reads_coarse": "cross", "joint": "joint"}
WHERE = ("uncertainty", "random", "grid", "oracle")
WHEN = ("attention", "entropy", "random", "grid")
WARMUP = ("none", "oracle", "random", "select")


@dataclass
class ModelConfig:
    backbone: str = "tiny"          # an OlmoEarth size (tiny, base, ...) or terramind_tiny[_joint]
    coarse_patch: int = 16          # S of the coarse tokens: 16, 14, 12, 10, 8, 6, 4 or 2 px
    coarse_time: str = "season"     # T of the coarse tokens: season composites | month (fixed
    #                                 monthly models: every month its own token)
    window_px: int = 8              # the selected windows' size; their fine tokens' S
    crop_time: str = "month"        # T of the fine tokens: month (cut from the monthly images) |
    #                                 season (cut from the coarse step's composite: space-only,
    #                                 one crop per (window, season))
    fine_patch: int = 0             # > 0: a window is tokenized as its sub-patches of this size
    #                                 (e.g. 14 px windows as 7 px tokens: S14Tsea+S7Tmon)
    attention: str = "one_way"      # one_way: fine tokens read the coarse ones and each other,
    #                                 coarse tokens never read fine ones (the coarse pass is
    #                                 reused); fine_reads_coarse: fine tokens read only the coarse
    #                                 ones; joint: one sequence of both
    coarse_head: bool = True        # the head writes a coarse_patch map per token (False: 16 x 16
    #                                 then pooled: the earlier fixed models' head)


@dataclass
class SelectConfig:
    """The window selector (mistr.core.selection.WindowSelector)."""
    where: str = "uncertainty"      # uncertainty | random | grid | oracle
    when: str = "attention"         # attention | entropy | random | grid (selection.py)
    percentage: float = 0.25        # q: the share of windows selected (where uncertainty: of the
    #                                 training distribution, one global threshold; else per sample)
    extra_steps: float = 0.0        # + that many extra (window, coarse step) pairs per selected
    #                                 window on average: the next step of the most uncertain ones
    # training draws extra_steps per batch from this comma list ("" = off): one model for
    # several time budgets (App. C)
    extra_steps_choices: str = ""
    # which step an extra pair adds to a picked window: the next most attended (attention) or the
    # one whose own coarse map disagrees most with the top step's (disagree)
    extra_pick: str = "attention"
    # where uncertainty: never pick a window with < half of its pixels holding S2 input (raw 0 in
    # every band in every valid month: YieldSAT outside the field polygon / tile padding); those
    # windows rank below every other in the threshold, so the budget moves to windows with data
    input_mask: bool = False


@dataclass
class BaselineConfig:
    """A published method to compare with, in place of the selector (mistr.baselines).
    Its tokens: model.coarse_patch at `tokens`; APT: coarse patches split to model.window_px."""
    method: str = "upsample"    # upsample | tldr | apt
    tokens: str = "seasons"     # seasons | months: what one token sequence step is
    upsampler: str = "anyup"    # upsample: bilinear | anyup | upa
    space: str = "features"     # upsample: features | predictions (the fixed model's probabilities
    #                             on the token grid; bilinear, upa; training-free)
    r: str = "0.25,0.085,0.043"     # tldr: tokens merged per block, shares of the input tokens
    keep: float = 0.5           # apt: share of coarse cells kept coarse (results/<task>/apt.json)
    scorer: str = "entropy"     # apt: entropy | laplacian (upstream's two patch scorers)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int | None = None      # None: the task's batch size
    seed: int = 0
    init_from: str = ""                # checkpoints/<name>.pt to start from (warm start)
    # warm-up: the first warmup_epochs train with warmup_windows windows of each sample picked by
    # the label oracle (or at random), every present month, attention warmup_attention; then the
    # selector, the optimizer restarted (backbone at lr x 0.1), the best val epoch from after
    warmup: str = "none"               # none | oracle | random | select
    # (select, the run's own selector with where = label oracle at select.percentage,
    # its "when" as configured (attention: one season per window); warmup_windows = 0.
    # warmup_epochs = epochs: a stage-1-only run, its LAST epoch saved -- shared by the stage-2
    # runs of every q of a resolution pair via train.init_from)
    warmup_epochs: int = 0
    warmup_windows: int = 0
    warmup_attention: str = "joint"
    head_only: bool = False            # backbone frozen (a linear probe: upsampler baselines)
    freeze_model: bool = False         # the model stays as loaded (training-free baselines)
    compute_split: str = "val"         # the split GMACs are traced on
    eval_percentages: str = ""         # after training, also test at each "q:extra" of this
    #                                    comma list (thresholds refit on val), one row each;
    #                                    q = the run's own percentage ("q:0.5,q:1")
    gpu_data: bool = True              # composite on the GPU (mistr.core.data)
    num_workers: int = 8
    max_hours: float | None = None     # stop before an epoch that would end past this; resume
    limit_batches: int = 0             # smoke test: a few batches, nothing written to results
    compute_batches: int = 4           # batches the reported GMACs/sample are traced on


NOT_HASHED = {"train.num_workers", "train.max_hours", "train.limit_batches",
              "train.compute_batches", "train.compute_split", "train.eval_percentages",
              "train.gpu_data"}


@dataclass
class RunConfig:
    task: str = "pastis"
    model: ModelConfig = field(default_factory=ModelConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fixed: bool = False                        # a fixed-resolution model: coarse tokens only
    baseline: BaselineConfig | None = None     # a published baseline instead of MiSTR

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fine_patch(self) -> int:
        return self.model.fine_patch or self.model.window_px

    def name(self) -> str:
        """A readable tag + a hash of every field off its default (see the module docstring)."""
        d = _off_default(self.to_dict(), asdict(RunConfig()))
        if self.baseline is not None:
            d["baseline"] = _off_default(asdict(self.baseline), asdict(BaselineConfig()))
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        m, s, t = self.model, self.select, self.train
        T = {"season": "sea", "month": "mon"}[m.coarse_time]
        coarse = f"S{m.coarse_patch}T{T}"
        if self.fixed:
            tag = f"fixed_{self.task}_{coarse}"
        elif self.baseline is not None:
            b = self.baseline
            tag = (f"{b.method}_{self.task}_S{m.coarse_patch}T{b.tokens[:3]}"
                   + {"upsample": f"-{b.upsampler}" + ("-pred" if b.space == "predictions" else "")
                      + ("-lp" if t.head_only else ""),
                      "tldr": f"-r{b.r}" + ("-ft" if t.epochs else ""),
                      "apt": f"-w{m.window_px}-keep{b.keep:g}"
                             + ("" if b.scorer == "entropy" else f"-{b.scorer}")}[b.method])
        else:
            tag = (f"mistr_{self.task}_{coarse}+S{self.fine_patch}T{m.crop_time[:3]}"
                   + (f"-w{m.window_px}" if m.fine_patch else "")
                   + f"_q{100 * s.percentage:g}"
                   + (f"_x{s.extra_steps:g}" if s.extra_steps else "")
                   + (f"_xset{s.extra_steps_choices.replace(',', '-')}" if s.extra_steps_choices else "")
                   + ("_xdis" if s.extra_pick == "disagree" else "")
                   + (f"_where-{s.where}" if s.where != "uncertainty" else "")
                   + ("_inmask" if s.input_mask else "")
                   + (f"_when-{s.when}" if s.when != "attention" else "")
                   + (f"_w{t.warmup}{t.warmup_epochs}" + ("-ow" if t.warmup_attention == "one_way" else "")
                      if t.warmup != "none" else "")
                   + (f"_{m.attention}" if m.attention != "one_way" else ""))
        tag += ("" if m.coarse_head else "_head16") \
            + (f"_{m.backbone}" if m.backbone != "tiny" else "") \
            + ("_warm" if t.init_from else "")
        return f"{tag}_ep{t.epochs}_seed{t.seed}_{h}"

    def check(self, task_kind: str):
        """Raise on a composition that cannot run."""
        m, s, t = self.model, self.select, self.train
        if m.attention not in ATTENTION or m.coarse_time not in ("season", "month"):
            raise ValueError(f"model.attention {m.attention!r} / coarse_time {m.coarse_time!r}")
        if m.coarse_patch not in (2, 4, 6, 8, 10, 12, 14, 16):
            raise ValueError(f"model.coarse_patch {m.coarse_patch}: 16, 14, 12, 10, 8, 6, 4 or 2")
        if t.warmup not in WARMUP or t.warmup_attention not in ATTENTION:
            raise ValueError(f"train.warmup {t.warmup!r} / warmup_attention {t.warmup_attention!r}")
        if self.fixed:
            if self.baseline is not None or t.warmup != "none" or t.freeze_model:
                raise ValueError("fixed: no baseline, warm-up or frozen model")
            return
        if m.crop_time not in ("month", "season"):
            raise ValueError(f"model.crop_time {m.crop_time!r}: month or season")
        if m.coarse_time == "month" and m.crop_time != "month":
            raise ValueError("model.coarse_time month: crops are the months' own (crop_time month)")
        if self.baseline is not None:
            b = self.baseline
            if (b.method not in ("upsample", "tldr", "apt") or b.tokens not in ("seasons", "months")
                    or b.upsampler not in ("bilinear", "anyup", "upa")
                    or b.space not in ("features", "predictions")
                    or (b.space == "predictions" and b.upsampler == "anyup")
                    or m.attention != "joint"):
                raise ValueError(f"baseline {b}: attention joint")
            if not t.init_from:
                raise ValueError("baseline: train.init_from, the fixed-resolution model it starts from")
            if t.compute_split not in ("val", "test"):
                raise ValueError(f"train.compute_split {t.compute_split!r}: val | test")
            if b.space == "predictions" and (t.epochs or not t.freeze_model):
                raise ValueError("training-free baseline: train.epochs=0 train.freeze_model=1")
            if b.method == "tldr" and bool(t.epochs) == t.freeze_model:
                raise ValueError("vid-TLDR: training-free (train.epochs=0 train.freeze_model=1) "
                                 "or fine-tuned with merging (epochs > 0, not frozen)")
            if b.scorer not in ("entropy", "laplacian"):
                raise ValueError(f"baseline.scorer {b.scorer!r}: entropy | laplacian")
            if b.method == "apt" and 2 * m.window_px != m.coarse_patch:
                raise ValueError("baseline apt: model.window_px = model.coarse_patch / 2")
            return
        if s.where not in WHERE or s.when not in WHEN:
            raise ValueError(f"select.where {s.where!r} / when {s.when!r}")
        if s.when == "entropy" and task_kind != "classification":
            raise ValueError("select.when entropy: classification tasks only")
        if (s.extra_steps or s.extra_steps_choices) and s.when != "attention":
            raise ValueError("select.extra_steps / extra_steps_choices: with when attention")
        if s.extra_pick not in ("attention", "disagree"):
            raise ValueError(f"select.extra_pick {s.extra_pick!r}: attention | disagree")
        xset = [float(v) for v in s.extra_steps_choices.split(",") if v]
        if not 0 < s.percentage <= 1 or s.extra_steps < 0 or (xset and min(xset) < 0):
            raise ValueError("select.percentage in (0, 1], extra_steps (+ choices) >= 0")
        if m.fine_patch and (m.fine_patch >= m.window_px or m.window_px % m.fine_patch):
            raise ValueError("model.fine_patch: a divisor of model.window_px, finer")
        if self.fine_patch >= m.coarse_patch:
            raise ValueError("the fine tokens must be finer than model.coarse_patch")
        if (t.warmup != "none") != (t.warmup_epochs > 0) or (t.warmup != "none" and not (
                t.warmup_epochs <= t.epochs and (t.warmup_windows > 0) != (t.warmup == "select")
                and not t.init_from)):
            raise ValueError("train.warmup: warmup_epochs in (0, epochs] (= epochs: stage 1 only), "
                             "warmup_windows > 0 (0 for select), no init_from")
        if t.freeze_model:
            raise ValueError("train.freeze_model: baselines only")


def _off_default(d: dict, dflt: dict, pre: str = "") -> dict:
    """The fields of d (nested) that differ from dflt, NOT_HASHED left out; task always in."""
    out = {}
    for k, v in d.items():
        p = pre + k
        if p in NOT_HASHED:
            continue
        dv = dflt.get(k) if isinstance(dflt, dict) else None
        if isinstance(v, dict) and isinstance(dv, dict):
            sub = _off_default(v, dv, p + ".")
            if sub:
                out[k] = sub
        elif v != dv or p == "task":
            out[k] = v
    return out


# options of the research code no paper run sets: an older config (a checkpoint's) may still carry
# them at these defaults, and loads; any other value cannot run here
REMOVED = {"ModelConfig": {"months_prior": False, "type_embedding": False, "tile_px": 64},
           "SelectConfig": {"percentage_choices": "", "min_pred": 0.0},
           "BaselineConfig": {"out_px": 1}}


def from_dict(cls, d: dict):
    """Nested dataclass from a (partial) dict; unknown keys raise."""
    kw = {}
    names = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        gone = REMOVED.get(cls.__name__, {})
        if k not in names and k in gone:
            if v != gone[k]:
                raise KeyError(f"{cls.__name__}.{k} = {v!r}: an option this code no longer has")
            continue
        if k not in names:
            raise KeyError(f"{cls.__name__} has no field {k!r}")
        sub = _dataclass_of(names[k].type)
        kw[k] = from_dict(sub, v) if sub is not None and isinstance(v, dict) else v
    return cls(**kw)


def _dataclass_of(tp):
    for c in (ModelConfig, SelectConfig, TrainConfig, BaselineConfig):
        if tp is c or (isinstance(tp, str) and c.__name__ in tp) or c in getattr(tp, "__args__", ()):
            return c
    return None


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """`select.percentage=0.5`, `train.warmup=random`, `baseline=none` (None). A section that is
    None is created with its defaults when a field below it is set."""
    d = cfg.to_dict()
    for o in overrides:
        key, _, raw = o.partition("=")
        if not _:
            raise ValueError(f"override {o!r}: want key=value")
        *path, last = key.split(".")
        node, cls = d, RunConfig
        for p in path:
            sub = _dataclass_of({f.name: f for f in fields(cls)}[p].type)
            if node.get(p) is None:
                node[p] = asdict(sub())
            node, cls = node[p], sub
        ftype = {f.name: f for f in fields(cls)}[last].type
        node[last] = _parse(raw, ftype)
    return from_dict(RunConfig, d)


def _parse(raw: str, tp):
    if raw.lower() in ("none", "null") and "None" in str(tp):     # only for optional fields
        return None
    sub = _dataclass_of(tp)
    if sub is not None:
        return asdict(sub(**json.loads(raw))) if raw.startswith("{") else asdict(sub())
    s = str(tp)
    if "bool" in s:
        return raw.lower() in ("1", "true", "yes")
    if "int" in s and "float" not in s:
        return int(raw)
    if "float" in s:
        return float(raw)
    return raw


def load(path) -> RunConfig:
    """A saved config (a JSON path or the dict itself)."""
    d = json.loads(open(path).read()) if not isinstance(path, dict) else path
    return from_dict(RunConfig, d)


__all__ = ["RunConfig", "ModelConfig", "SelectConfig", "TrainConfig", "BaselineConfig",
           "from_dict", "apply_overrides", "load", "replace"]
