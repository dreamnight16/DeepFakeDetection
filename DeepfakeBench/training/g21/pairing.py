"""Content-first FF++ lineage pairing: choose A or B from images, with an audit.

The fake key ``A_B`` names two candidate real videos.  The officially documented
target/source order is only a hint: the extracted-frame index set is a pure
video-length fingerprint and is heavily confounded, so it can never verify a
mapping on its own.  This module therefore compares the fake's *background*
against both candidate originals, keeps the length/grid signal only as a
candidate hint and a timeline-compatibility check, and reports every
disagreement instead of hiding it.

Only a decisive, close content win with a compatible timeline yields a verified
rule.  Everything else becomes an explicit exclusion: nothing is guessed, no
frame is padded, and no method is silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath
import statistics

from .builder import METHODS, mapping_template
from .data import uniform_indices
from .io import atomic_json
from .manifest import image_path, relative_path

EPS = 1e-6
SAMPLE_MIN = 8
SAMPLE_MAX = 16

MASK_MODE_MANIPULATION = "manipulation_mask"
MASK_MODE_FALLBACK = "central_face_box_fallback"

DECISION_VERIFIED = "verified"
DECISION_AMBIGUOUS = "ambiguous_content"
DECISION_WEAK_MARGIN = "weak_margin"
DECISION_NOT_CLOSER = "not_closer"
DECISION_TIMELINE = "timeline_mismatch"
DECISION_INSUFFICIENT = "insufficient_intersection"
DECISION_MISSING = "missing_candidate"
DECISION_UNSUPPORTED = "unsupported_key"
DECISION_UNUSABLE = "unusable_frames"
DECISION_MALFORMED = "malformed_entry"

# How the two candidates were compared on this video.
MODE_HEAD_TO_HEAD = "head_to_head"
MODE_CONFIRMATION = "single_side_confirmation"

# Which candidate the frames the two sides *share* point at.  ``None`` means no
# such frame exists, i.e. there is no comparative evidence at all; SHARED_TIE is
# an exact dead heat over those frames, which is not evidence for either side.
SHARED_TIE = -1

# Fraction of measured rows that may die on ``eligible_floor`` before the run
# report tells the operator to check mask polarity instead of believing the
# ``unusable_frames`` bucket.
MASK_SUSPECT_FRACTION = 0.25

# Every machine-readable outcome, so the funnel schema never shifts between runs.
DECISION_CODES = (
    DECISION_VERIFIED,
    DECISION_AMBIGUOUS,
    DECISION_WEAK_MARGIN,
    DECISION_NOT_CLOSER,
    DECISION_TIMELINE,
    DECISION_INSUFFICIENT,
    DECISION_MISSING,
    DECISION_UNSUPPORTED,
    DECISION_UNUSABLE,
    DECISION_MALFORMED,
)

DISTANCE_EDGES = (0, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 257)
MARGIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0000001)


@dataclass(frozen=True)
class PairingConfig:
    """Thresholds fixed by the project owner; ``sample_frames`` is clamped 8..16."""

    sample_frames: int = 12
    # A decision may never rest on fewer same-index frames than this: two
    # medians computed over three readable frames each is not evidence.
    min_verification_frames: int = 8
    comparison_size: int = 224
    blur_sigma: float = 1.0
    mask_dilation: int = 8
    eligible_floor: float = 0.10
    face_box: tuple[float, float, float, float] = (0.25, 0.75, 0.15, 0.85)
    win_rate_min: float = 0.9
    margin_min: float = 0.5
    max_winner_distance: float = 12.0
    # A single-sided confirmation has no comparative evidence to fall back on,
    # so it demands a much closer absolute match than a head-to-head win.
    max_single_side_distance: float = 4.0
    timeline_jaccard_min: float = 0.9

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_frames",
            min(max(int(self.sample_frames), SAMPLE_MIN), SAMPLE_MAX),
        )
        object.__setattr__(
            self,
            "min_verification_frames",
            min(max(int(self.min_verification_frames), 2), self.sample_frames),
        )
        if type(self.comparison_size) is not int or self.comparison_size < 16:
            raise ValueError("comparison_size must be an integer >= 16")
        if not self.blur_sigma > 0:
            raise ValueError("blur_sigma must be positive")
        if type(self.mask_dilation) is not int or self.mask_dilation < 0:
            raise ValueError("mask_dilation must be a nonnegative integer")
        if not 0 < self.eligible_floor < 1:
            raise ValueError("eligible_floor must lie in (0, 1)")
        x0, x1, y0, y1 = self.face_box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("face_box must be [x0, x1, y0, y1] inside the frame")
        if not 0 <= self.win_rate_min <= 1:
            raise ValueError("win_rate_min must lie in [0, 1]")
        if self.margin_min < 0:
            raise ValueError("margin_min must be nonnegative")
        if self.max_winner_distance <= 0:
            raise ValueError("max_winner_distance must be positive")
        if not 0 < self.max_single_side_distance <= self.max_winner_distance:
            raise ValueError(
                "max_single_side_distance must be positive and no larger than "
                "max_winner_distance"
            )
        if not 0 <= self.timeline_jaccard_min <= 1:
            raise ValueError("timeline_jaccard_min must lie in [0, 1]")


def _rel(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected a frame path string: {value!r}")
    return relative_path(value.replace("\\", "/"))


def _frame_index(value: object) -> int:
    return int(PurePosixPath(_rel(value)).stem)


def frame_indices(entry: dict) -> list[int]:
    """Original frame indices, ascending and unique.

    *Stored order is deliberately not validated.*  The shipped FF++ index lists
    69 entries (2% of the real videos) whose frames are a permutation of a valid
    ascending run, and rejecting those was measured to cost 184 of 2877 training
    keys - 6.4% of the corpus - for no safety gain at all: a frame index denotes
    the same moment whatever position it occupies, and both the measurement here
    and the builder's ``real.keys() & fake.keys()`` address frames by that value,
    never by position.  ``frames_out_of_order`` records every entry that relied on
    this, and duplicates are still refused outright because there the value
    itself is ambiguous.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("frames"), list):
        raise ValueError("entry must carry a frames list")
    out: list[int] = []
    seen: set[int] = set()
    for raw in entry["frames"]:
        index = _frame_index(raw)
        if index in seen:
            raise ValueError("frame indices must be unique")
        seen.add(index)
        out.append(index)
    if not out:
        raise ValueError("entry has no frames")
    return sorted(out)


def frame_order_is_ascending(entry: dict) -> bool:
    """Whether the stored list is already ascending, i.e. the seen-order flag."""
    stored = [_frame_index(raw) for raw in entry["frames"]]
    return stored == sorted(stored)


def frame_slots(entry: dict) -> dict[int, int]:
    """Map an original frame index to its position in the *stored* list.

    Positions, not the sorted indices, are what indexes ``entry["frames"]``, so
    this is the mapping the reader needs - and it is order-independent, which is
    exactly why an out-of-order list is safe to accept.
    """
    return {_frame_index(raw): slot for slot, raw in enumerate(entry["frames"])}


def subsample(values: list[int], count: int) -> list[int]:
    """Uniformly subsample a sorted list, preserving its order."""
    if len(values) <= count:
        return list(values)
    return [values[position] for position in uniform_indices(len(values), count)]


def intersect_indices(
    fake_indices: list[int], candidate_indices: list[int]
) -> list[int]:
    """The frames the builder will actually pair: identical original index.

    ``build_ffpp`` turns ``same_original_index_verified`` into
    ``matches = [[i, i] for i in real.keys() & fake.keys()]``, so the only
    honest support for that assertion is this intersection - measured here,
    frame for frame, at the very indices it will train on.  Approximate
    alignments (normalized-time, neighbour substitution) are deliberately not
    used: the design forbids asserting an index correspondence that the
    evidence did not establish.
    """
    return sorted(set(fake_indices) & set(candidate_indices))


def _read_gray(root: str | Path, relative: object, config: PairingConfig):
    import cv2
    import numpy as np

    path = image_path(root, _rel(relative))
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise ValueError(f"cannot decode image: {relative}")
    return cv2.resize(
        image,
        (config.comparison_size, config.comparison_size),
        interpolation=cv2.INTER_AREA,
    )


def _blurred(gray, config: PairingConfig):
    import cv2

    return cv2.GaussianBlur(
        gray, (0, 0), sigmaX=config.blur_sigma, sigmaY=config.blur_sigma
    ).astype("float32")


def _dilate(flags, config: PairingConfig):
    import cv2
    import numpy as np

    if config.mask_dilation == 0:
        return flags
    size = 2 * config.mask_dilation + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(flags.astype(np.uint8), kernel, iterations=1).astype(bool)


def _mask_flags(
    root: str | Path, entry: dict, position: int, config: PairingConfig
) -> tuple[object, str]:
    """Exclusion flags for one fake frame: the FF++ mask, or a central face box.

    The mask is dilated outward because swap edges, hair and fusion boundaries
    contaminate the background.  If the mask is missing or unreadable the face
    box is used instead and the fallback is recorded, never hidden.
    """
    import cv2
    import numpy as np

    size = config.comparison_size
    masks = entry.get("masks")
    if (
        isinstance(masks, list)
        and 0 <= position < len(masks)
        and isinstance(masks[position], str)
    ):
        try:
            path = image_path(root, _rel(masks[position]))
            image = cv2.imdecode(
                np.frombuffer(path.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
        except (OSError, ValueError):
            image = None
        if image is not None:
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
            return resized > 127, MASK_MODE_MANIPULATION
    x0, x1, y0, y1 = config.face_box
    box = np.zeros((size, size), dtype=bool)
    box[int(y0 * size) : int(y1 * size), int(x0 * size) : int(x1 * size)] = True
    return box, MASK_MODE_FALLBACK


def _eligible(flags, config: PairingConfig):
    return ~_dilate(flags, config)


def _background_distance(fake_blur, candidate_blur, eligible) -> float:
    import numpy as np

    if not bool(eligible.any()):
        raise ValueError("eligible region is empty")
    distance = float(np.abs(fake_blur - candidate_blur)[eligible].mean())
    if not math.isfinite(distance):
        # A non-finite distance would silently poison the medians and then abort
        # the JSON write with no audit at all; refuse the frame instead.
        raise ValueError("background distance is not finite")
    return distance


def _trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) >= 3:
        ordered = ordered[1:-1]
    return math.fsum(ordered) / len(ordered)


def _decision_record(
    method: str, key: str, ids: list[str], decision: str, **extra
) -> dict:
    record = {
        "method": method,
        "fake_key": key,
        "candidate_a": ids[0] if len(ids) > 0 else None,
        "candidate_b": ids[1] if len(ids) > 1 else None,
        "reference_component": None,
        "winner": None,
        "mode": None,
        "d_A": None,
        "d_B": None,
        "median_A": None,
        "median_B": None,
        "trimmed_mean_A": None,
        "trimmed_mean_B": None,
        "win_rate": None,
        "margin_ratio": None,
        "eligible_fraction": None,
        "mask_modes": [],
        "estimated_lengths": {},
        "intersection_A": 0,
        "intersection_B": 0,
        "intersection_common": 0,
        "measured_A": 0,
        "measured_B": 0,
        "confirmed_A": False,
        "confirmed_B": False,
        "shared_pairs": 0,
        "shared_agreement": None,
        "frames_out_of_order": [],
        "timeline_jaccard": None,
        "timeline_coverage": None,
        "timelines_match": False,
        "frame_pairs": [],
        "decision": decision,
        "verified": False,
        "error": None,
    }
    record.update(extra)
    return record


def _measure_frames(
    root: str | Path,
    fake_entry: dict,
    fake_slot: dict[int, int],
    a_entry: dict,
    a_slot: dict[int, int],
    b_entry: dict,
    b_slot: dict[int, int],
    a_lookup: dict[int, int],
    b_lookup: dict[int, int],
    config: PairingConfig,
) -> list[dict]:
    """Background distance per requested fake index, for each candidate that has it.

    ``a_lookup``/``b_lookup`` map a fake index to the candidate frame index it is
    compared against.  A missing key means that candidate has no frame at that
    original index, so it contributes nothing to that row rather than being
    substituted with a neighbouring frame.
    """
    rows: list[dict] = []
    for fake_index in sorted(set(a_lookup) | set(b_lookup)):
        position = fake_slot[fake_index]
        flags, mask_mode = _mask_flags(root, fake_entry, position, config)
        eligible = _eligible(flags, config)
        fraction = float(eligible.sum()) / float(eligible.size)
        a_index = a_lookup.get(fake_index)
        b_index = b_lookup.get(fake_index)
        row = {
            "fake_index": fake_index,
            "A_index": a_index,
            "B_index": b_index,
            "d_A": None,
            "d_B": None,
            "eligible_fraction": fraction,
            "mask_mode": mask_mode,
            "usable": False,
            "note": None,
        }
        if fraction < config.eligible_floor:
            row["note"] = "eligible_fraction_below_floor"
            rows.append(row)
            continue
        try:
            fake_blur = _blurred(
                _read_gray(root, fake_entry["frames"][position], config), config
            )
        except (OSError, ValueError):
            row["note"] = "fake_frame_unreadable"
            rows.append(row)
            continue
        notes: list[str] = []
        if a_index is not None:
            try:
                candidate_blur = _blurred(
                    _read_gray(root, a_entry["frames"][a_slot[a_index]], config), config
                )
                row["d_A"] = _background_distance(fake_blur, candidate_blur, eligible)
            except (OSError, ValueError):
                notes.append("candidate_A_frame_unreadable")
        if b_index is not None:
            try:
                candidate_blur = _blurred(
                    _read_gray(root, b_entry["frames"][b_slot[b_index]], config), config
                )
                row["d_B"] = _background_distance(fake_blur, candidate_blur, eligible)
            except (OSError, ValueError):
                notes.append("candidate_B_frame_unreadable")
        row["note"] = ";".join(notes) if notes else None
        row["usable"] = row["d_A"] is not None and row["d_B"] is not None
        rows.append(row)
    return rows


def decide_video(
    method: str,
    key: str,
    fake_entry: dict,
    a_entry: dict | None,
    b_entry: dict | None,
    *,
    root: str | Path,
    config: PairingConfig = PairingConfig(),
) -> dict:
    """Decide A vs B for one fake video and return a fully populated audit record."""
    ids = key.split("_")
    if len(ids) != 2 or not all(x.isdigit() for x in ids):
        return _decision_record(method, key, ids[:2], DECISION_UNSUPPORTED)
    if a_entry is None or b_entry is None:
        missing = [i for i, entry in zip(ids, (a_entry, b_entry)) if entry is None]
        return _decision_record(
            method, key, ids, DECISION_MISSING, missing_candidates=missing
        )
    try:
        return _decide_present_candidates(
            method, key, ids, fake_entry, a_entry, b_entry, root=root, config=config
        )
    except (ValueError, OSError, KeyError) as exc:
        # One malformed entry must never abort the sweep: the contract is that
        # every key ends up either verified or explicitly excluded.
        return _decision_record(
            method, key, ids, DECISION_MALFORMED, error=f"{type(exc).__name__}: {exc}"
        )


def _decide_present_candidates(
    method: str,
    key: str,
    ids: list[str],
    fake_entry: dict,
    a_entry: dict,
    b_entry: dict,
    *,
    root: str | Path,
    config: PairingConfig,
) -> dict:
    fake_indices = frame_indices(fake_entry)
    a_indices = frame_indices(a_entry)
    b_indices = frame_indices(b_entry)
    out_of_order = [
        name
        for name, entry in (("fake", fake_entry), ("A", a_entry), ("B", b_entry))
        if not frame_order_is_ascending(entry)
    ]
    fake_slot = frame_slots(fake_entry)
    a_slot = frame_slots(a_entry)
    b_slot = frame_slots(b_entry)

    inter_a = intersect_indices(fake_indices, a_indices)
    inter_b = intersect_indices(fake_indices, b_indices)
    common = sorted(set(inter_a) & set(inter_b))
    floor = config.min_verification_frames

    # Both candidates can be measured on the same fake frames -> a fair
    # head-to-head.  Otherwise each side is judged on the frames it shares with
    # the fake, and exactly one decisive, uncontradicted side names the
    # reference; any frame both sides *do* share is still weighed (see
    # ``shared_agreement``), it is only too few for a full head-to-head.
    if len(common) >= floor:
        mode = MODE_HEAD_TO_HEAD
        shared = subsample(common, config.sample_frames)
        a_lookup = {index: index for index in shared}
        b_lookup = dict(a_lookup)
    else:
        mode = MODE_CONFIRMATION
        a_lookup = {index: index for index in subsample(inter_a, config.sample_frames)}
        b_lookup = {index: index for index in subsample(inter_b, config.sample_frames)}
    frames = _measure_frames(
        root, fake_entry, fake_slot, a_entry, a_slot, b_entry, b_slot,
        a_lookup, b_lookup, config,
    )

    a_values = [row["d_A"] for row in frames if row["d_A"] is not None]
    b_values = [row["d_B"] for row in frames if row["d_B"] is not None]
    paired = [
        (row["d_A"], row["d_B"])
        for row in frames
        if row["d_A"] is not None and row["d_B"] is not None
    ]
    median_a = statistics.median(a_values) if a_values else None
    median_b = statistics.median(b_values) if b_values else None
    trimmed_a = _trimmed_mean(a_values) if a_values else None
    trimmed_b = _trimmed_mean(b_values) if b_values else None

    # A median over three surviving frames is not evidence: a decision may never
    # rest on fewer than ``min_verification_frames`` measured frames.  A
    # head-to-head also requires the winner to beat the loser by both statistics;
    # a side confirmed on its own frames instead has to clear the much tighter
    # ``max_single_side_distance``, because the second opinion that catches a
    # merely close-by scene is only available on the frames the two sides share.
    bound = (
        config.max_winner_distance
        if mode == MODE_HEAD_TO_HEAD
        else config.max_single_side_distance
    )
    enough_a = len(a_values) >= floor
    enough_b = len(b_values) >= floor
    close_a = (
        median_a is not None
        and trimmed_a is not None
        and max(median_a, trimmed_a) <= bound
    )
    close_b = (
        median_b is not None
        and trimmed_b is not None
        and max(median_b, trimmed_b) <= bound
    )
    confirmed_a = enough_a and close_a
    confirmed_b = enough_b and close_b

    # Any frame the two candidates *share* is comparative evidence and stays
    # decisive even in single-side mode.  Confirming a side on its own absolute
    # distance alone let a merely near-by wrong scene win whenever the true
    # source happened to be measurable on fewer frames; the shared median is the
    # one comparison that can outvote that, so a side it contradicts - or merely
    # ties - is not a confirmation.
    if paired:
        shared_median_a = statistics.median([value_a for value_a, _ in paired])
        shared_median_b = statistics.median([value_b for _, value_b in paired])
        if shared_median_a < shared_median_b:
            shared_agreement = 0
        elif shared_median_b < shared_median_a:
            shared_agreement = 1
        else:
            shared_agreement = SHARED_TIE
    else:
        shared_agreement = None
    corroborated_a = confirmed_a and shared_agreement in (None, 0)
    corroborated_b = confirmed_b and shared_agreement in (None, 1)

    winner_trimmed = _argmin_two(trimmed_a, trimmed_b)
    if mode == MODE_HEAD_TO_HEAD:
        winner_median = _argmin_two(median_a, median_b)
    elif corroborated_a and not corroborated_b:
        winner_median = 0
    elif corroborated_b and not corroborated_a:
        winner_median = 1
    else:
        winner_median = None

    win_rate = None
    margin_ratio = None
    if winner_median is not None and paired:
        wins = sum(
            (value_a < value_b) if winner_median == 0 else (value_b < value_a)
            for value_a, value_b in paired
        )
        win_rate = wins / len(paired)
    # A margin across two *different* frame supports compares medians that do not
    # stand on the same footing and reads as decisive in whichever direction the
    # larger support happens to point, so it is reported only when both
    # candidates were measured on exactly the same frames.
    measured_a_on = {row["fake_index"] for row in frames if row["d_A"] is not None}
    measured_b_on = {row["fake_index"] for row in frames if row["d_B"] is not None}
    if (
        measured_a_on
        and measured_a_on == measured_b_on
        and median_a is not None
        and median_b is not None
        and median_a != median_b
    ):
        d_winner = min(median_a, median_b)
        d_loser = max(median_a, median_b)
        margin_ratio = (d_loser - d_winner) / (d_loser + EPS)

    if mode == MODE_HEAD_TO_HEAD:
        if not a_values or not b_values or len(paired) < floor:
            decision = DECISION_UNUSABLE
        elif winner_median is None or winner_trimmed != winner_median:
            decision = DECISION_AMBIGUOUS
        elif win_rate is not None and win_rate < config.win_rate_min:
            decision = DECISION_AMBIGUOUS
        elif margin_ratio is not None and margin_ratio < config.margin_min:
            decision = DECISION_WEAK_MARGIN
        elif (winner_median == 0 and not close_a) or (
            winner_median == 1 and not close_b
        ):
            # The margin ratio alone is fooled when BOTH candidates are far away
            # (d_A=100, d_B=180 still gives r=0.29 and looks decisive); the
            # absolute closeness guard rejects an unverifiable "winner".
            decision = DECISION_NOT_CLOSER
        else:
            decision = DECISION_VERIFIED
    elif confirmed_a and confirmed_b:
        # Both sides match the fake on their own frames but share too few frames
        # for a head-to-head: refuse rather than pick on incomparable supports.
        decision = DECISION_AMBIGUOUS
    elif corroborated_a != corroborated_b:
        # Exactly one side cleared the tight absolute bound *and* is not
        # contradicted by the frames both sides share.  The sole no-shared-frame
        # case (common == 0, the FF-FS/NT shape) has nothing to contradict it,
        # which is why this mode's absolute bound is 4.0 and not 12.0.
        decision = DECISION_VERIFIED
    elif not a_values and not b_values:
        decision = DECISION_UNUSABLE
    elif not enough_a or not enough_b:
        decision = DECISION_INSUFFICIENT
    else:
        decision = DECISION_NOT_CLOSER

    timeline_jaccard = None
    timeline_coverage = None
    timelines_match = False
    if winner_median is not None:
        winner_indices = a_indices if winner_median == 0 else b_indices
        timeline_jaccard = _jaccard(fake_indices, winner_indices)
        # Reported next to the gate so a video dropped for a *nested* grid (every
        # fake index present, but the winner extracted more frames) is
        # distinguishable from one dropped for a genuinely disjoint timeline.
        timeline_coverage = len(set(fake_indices) & set(winner_indices)) / len(
            set(fake_indices)
        )
        timelines_match = timeline_jaccard >= config.timeline_jaccard_min
        if decision == DECISION_VERIFIED and not timelines_match:
            # Content says A/B but the sampled grid does not line up; the design
            # forbids papering over misalignment with neighbour/time-warping
            # rules, so the video is excluded for a human to override.
            decision = DECISION_TIMELINE

    verified = decision == DECISION_VERIFIED
    mask_modes = sorted({row["mask_mode"] for row in frames})
    eligible_fraction = (
        statistics.median([row["eligible_fraction"] for row in frames])
        if frames
        else None
    )
    return _decision_record(
        method,
        key,
        ids,
        decision,
        reference_component=winner_median,
        winner=_component_name(winner_median),
        mode=mode,
        d_A=median_a,
        d_B=median_b,
        median_A=median_a,
        median_B=median_b,
        trimmed_mean_A=trimmed_a,
        trimmed_mean_B=trimmed_b,
        win_rate=win_rate,
        margin_ratio=margin_ratio,
        eligible_fraction=eligible_fraction,
        mask_modes=mask_modes,
        estimated_lengths={
            "fake": len(fake_indices),
            "A": len(a_indices),
            "B": len(b_indices),
        },
        intersection_A=len(inter_a),
        intersection_B=len(inter_b),
        intersection_common=len(common),
        measured_A=len(a_values),
        measured_B=len(b_values),
        confirmed_A=confirmed_a,
        confirmed_B=confirmed_b,
        shared_pairs=len(paired),
        shared_agreement=shared_agreement,
        frames_out_of_order=out_of_order,
        timeline_jaccard=timeline_jaccard,
        timeline_coverage=timeline_coverage,
        timelines_match=timelines_match,
        frame_pairs=frames,
        verified=verified,
    )


def _argmin_two(first: float | None, second: float | None) -> int | None:
    if first is None or second is None or first == second:
        return None
    return 0 if first < second else 1


def _component_name(component: int | None) -> str | None:
    if component is None:
        return None
    return "A" if component == 0 else "B"


def _jaccard(first: list[int], second: list[int]) -> float:
    left, right = set(first), set(second)
    union = left | right
    if not union:
        raise ValueError("timeline comparison needs nonempty index sets")
    return len(left & right) / len(union)


def _fixed(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _evidence(record: dict) -> str:
    # win_rate is legitimately absent when a single side was confirmed without
    # any jointly observable frame, so every number is formatted defensively.
    return (
        f"g21 content pairing ({record['mode']}): "
        f"median_A={_fixed(record['median_A'], 4)} "
        f"median_B={_fixed(record['median_B'], 4)} "
        f"win_rate={_fixed(record['win_rate'], 4)} "
        f"margin_ratio={_fixed(record['margin_ratio'], 4)} "
        f"measured_A={record['measured_A']} measured_B={record['measured_B']} "
        f"common={record['intersection_common']} "
        f"shared={record['shared_pairs']} "
        f"shared_agreement={record['shared_agreement']} "
        f"timeline_jaccard={_fixed(record['timeline_jaccard'], 6)} "
        f"timeline_coverage={_fixed(record['timeline_coverage'], 4)}"
    )


def _exclude_detail(record: dict) -> dict:
    return {
        "mode": record["mode"],
        "d_A": record["d_A"],
        "d_B": record["d_B"],
        "win_rate": record["win_rate"],
        "margin_ratio": record["margin_ratio"],
        "eligible_fraction": record["eligible_fraction"],
        "mask_modes": record["mask_modes"],
        "estimated_lengths": record["estimated_lengths"],
        "intersection_A": record["intersection_A"],
        "intersection_B": record["intersection_B"],
        "intersection_common": record["intersection_common"],
        "measured_A": record["measured_A"],
        "measured_B": record["measured_B"],
        "confirmed_A": record["confirmed_A"],
        "confirmed_B": record["confirmed_B"],
        "shared_pairs": record["shared_pairs"],
        "shared_agreement": record["shared_agreement"],
        "timeline_jaccard": record["timeline_jaccard"],
        "timeline_coverage": record["timeline_coverage"],
        "timelines_match": record["timelines_match"],
        "error": record["error"],
    }


def _override(record: dict) -> dict:
    if record["verified"]:
        # same_original_index_verified is claimable precisely because the
        # distances above were measured on this very intersection: the frames
        # the builder will pair are the frames the evidence covers.
        return {
            "verified": True,
            "reference_component": record["reference_component"],
            "same_original_index_verified": True,
            "evidence": _evidence(record),
        }
    return {
        "excluded": True,
        "exclude_reason": record["decision"],
        "exclude_detail": _exclude_detail(record),
    }


def _funnel_bucket() -> dict:
    # A partition of every fake video seen: the named buckets plus "other" sum
    # to "total".  The full breakdown lives in decision_counts.
    return {
        "total": 0,
        "verified": 0,
        "ambiguous_content": 0,
        "weak_margin": 0,
        "not_closer": 0,
        "timeline_mismatch": 0,
        "insufficient_intersection": 0,
        "other": 0,
        "usable_pairs": 0,
    }


def _empty_funnel() -> dict:
    return {"overall": _funnel_bucket(), "by_method": {m: _funnel_bucket() for m in METHODS}}


def _record_funnel(funnel: dict, method: str, decision: str) -> None:
    named = (
        DECISION_VERIFIED,
        DECISION_AMBIGUOUS,
        DECISION_WEAK_MARGIN,
        DECISION_NOT_CLOSER,
        DECISION_TIMELINE,
        DECISION_INSUFFICIENT,
    )
    bucket_key = decision if decision in named else "other"
    for bucket in (funnel["overall"], funnel["by_method"][method]):
        bucket["total"] += 1
        bucket[bucket_key] += 1
        if decision == DECISION_VERIFIED:
            bucket["usable_pairs"] += 1


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty distribution")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _finite(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _distribution(values: list[float], edges: tuple[float, ...]) -> dict:
    # Non-finite values cannot be binned and would break the JSON write; they are
    # dropped here so the histogram stays a histogram.
    ordered = sorted(float(value) for value in values if _finite(value))
    histogram = [
        {"lo": edges[i], "hi": edges[i + 1], "count": 0} for i in range(len(edges) - 1)
    ]
    above = 0
    for value in ordered:
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                histogram[i]["count"] += 1
                break
        else:
            above += 1
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "quantiles": {},
            "histogram": histogram,
            "above_last_edge": 0,
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
        "median": _quantile(ordered, 0.5),
        "quantiles": {
            f"{q:g}": _quantile(ordered, q)
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "histogram": histogram,
        "above_last_edge": above,
    }


def _distributions(decisions: list[dict]) -> dict:
    """Histograms over *verified* decisions, plus the rejected side of each.

    Mixing refused records in would put "the reference we accepted" and "the
    candidate we turned down" in one histogram and hide the very bimodality the
    operator calibrates the thresholds from, so a rejected record contributes
    nothing here at all.  ``d_rejected_on_verified`` is the losing candidate's
    distance on the same verified videos, so the two are directly comparable.
    """
    winners: list[float] = []
    losers: list[float] = []
    margins: list[float] = []
    for record in decisions:
        if not record["verified"]:
            continue
        winner = record["winner"]
        d_winner = record["d_A"] if winner == "A" else record["d_B"]
        d_loser = record["d_B"] if winner == "A" else record["d_A"]
        if _finite(d_winner):
            winners.append(d_winner)
        if _finite(d_loser):
            losers.append(d_loser)
        if _finite(record["margin_ratio"]):
            margins.append(record["margin_ratio"])
    return {
        "d_winner_verified": _distribution(winners, DISTANCE_EDGES),
        "d_rejected_on_verified": _distribution(losers, DISTANCE_EDGES),
        "margin_ratio_verified": _distribution(margins, MARGIN_EDGES),
    }


def _warnings(decisions: list[dict]) -> list[str]:
    """Run-level symptoms no single record can show."""
    rows = [row for record in decisions for row in record["frame_pairs"]]
    warnings: list[str] = []
    if rows:
        starved = sum(
            1 for row in rows if row["note"] == "eligible_fraction_below_floor"
        )
        if starved / len(rows) >= MASK_SUSPECT_FRACTION:
            warnings.append(
                f"{starved}/{len(rows)} measured rows had an eligible fraction "
                "below eligible_floor: check mask polarity and face_box before "
                "reading unusable_frames as 'no background available' - an "
                "inverted mask looks exactly like this"
            )
    if decisions and not any(
        MASK_MODE_MANIPULATION in record["mask_modes"] for record in decisions
    ):
        warnings.append(
            "no readable manipulation mask in the whole run; every distance fell "
            "back to the central face box, which is weaker evidence"
        )
    displaced = sum(1 for record in decisions if record["frames_out_of_order"])
    if displaced:
        warnings.append(
            f"{displaced} of {len(decisions)} keys referenced a frame list stored "
            "out of ascending order (an FF++ index quirk, 69 real videos); they "
            "were accepted because a frame index is addressed by value, not by "
            "position - verify this against the extraction if the corpus changes"
        )
    blind = [
        record
        for record in decisions
        if record["verified"] and record["shared_pairs"] == 0
    ]
    if blind:
        # The one hole the shared-frame rule structurally cannot close: with no
        # frame measurable on both candidates there is nothing to contradict the
        # single-sided bound, and two candidates whose backgrounds differ by a
        # couple of gray levels are indistinguishable by any background test.
        # Named explicitly so those keys can be reviewed by hand.
        warnings.append(
            f"{len(blind)} verified videos had no frame measurable on both "
            "candidates, so they rest on the single-sided absolute bound alone "
            "(max_single_side_distance); list them from the audit and review "
            "those keys by hand"
        )
    return warnings


def run_pairing(
    dataset_json: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    config: PairingConfig = PairingConfig(),
) -> dict:
    """Pair every FF++ train fake with its content reference and write the audit.

    Writes ``pair_map.json`` (consumable by ``builder.build_ffpp``) and
    ``pairing_audit.json``.  Every key receives an override: a verified rule, or
    an explicit exclusion.  A method whose every key is excluded will still fail
    loudly in the builder rather than silently train on fewer methods.
    """
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))[
        "FaceForensics++"
    ]
    real_train = data["FF-real"]["train"]["c23"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"pairing output already exists: {output}")

    decisions: list[dict] = []
    overrides: dict[str, dict] = {}
    funnel = _empty_funnel()
    references: dict[str, set[str]] = {method: set() for method in METHODS}
    for method in METHODS:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            ids = key.split("_")
            valid = len(ids) == 2 and all(x.isdigit() for x in ids)
            a_entry = real_train.get(ids[0]) if valid else None
            b_entry = real_train.get(ids[1]) if valid else None
            record = decide_video(
                method, key, entry, a_entry, b_entry, root=data_root, config=config
            )
            decisions.append(record)
            overrides[f"{method}/{key}"] = _override(record)
            _record_funnel(funnel, method, record["decision"])
            if record["verified"]:
                reference = record[("candidate_a", "candidate_b")[record["reference_component"]]]
                if isinstance(reference, str):
                    references[method].add(reference)

    template = mapping_template()
    template["overrides"] = overrides
    output.mkdir(parents=True)
    atomic_json(output / "pair_map.json", template)
    audit = {
        "schema_version": 1,
        "config": asdict(config),
        "decision_counts": {
            code: sum(r["decision"] == code for r in decisions) for code in DECISION_CODES
        },
        "funnel": funnel,
        "warnings": _warnings(decisions),
        "distributions": _distributions(decisions),
        "distinct_references": {
            method: sorted(references[method]) for method in METHODS
        },
        "references_by_method": {
            method: len(references[method]) for method in METHODS
        },
        "decisions": decisions,
    }
    atomic_json(output / "pairing_audit.json", audit)
    return {
        "schema_version": 1,
        "usable_pairs": funnel["overall"]["usable_pairs"],
        "decision_counts": audit["decision_counts"],
        "funnel": funnel,
        "warnings": audit["warnings"],
        "distributions": audit["distributions"],
        "references_by_method": audit["references_by_method"],
    }
