#!/usr/bin/env python3
"""
Rover Navigation Simulator
===========================

A planetary rover receives a sequence of movement commands while exploring
unknown terrain. This module computes the rover's resulting position and
orientation after each command using homogeneous-coordinate transformation
matrices, records the full trajectory, and renders it with Matplotlib.

Architecture (mirrors the project design document):

    Input Commands -> Command Parser -> Transformation Generator ->
    Matrix Operations Engine -> Rover State Update -> Trajectory Storage ->
    Visualization

Transformation convention
--------------------------
Every command is expressed as a transformation matrix in the ROVER'S LOCAL
FRAME (forward = local +x axis, rotation = about the rover's own origin).
Each local transform is composed onto the rover's accumulated world-frame
transform via right-multiplication:

    W_n = W_(n-1) @ T_local_n

This is standard forward-kinematics composition and guarantees that
sequential, per-step state updates are mathematically identical to applying
a single composite matrix built by multiplying the same local transforms in
chronological order (verified in TestChainedTransformations below).

Manual configuration
---------------------
Two modes are supported, both fully runtime-configurable, never hardcoded:

1. Point-to-point mode (default): the program asks for a start point and a
   target end point (interactively, or via --start-x/--start-y/--start-heading
   and --end-x/--end-y/--end-heading), derives the rotate-then-drive command
   sequence that connects them (TrajectoryPlanner), and shows the full
   geometric derivation alongside the matrix arithmetic.
2. Explicit command mode (--commands / --commands-file): the classic mode
   where the exact command sequence is supplied directly, preserved for
   scripting and for the automated test suite.

Matrix calculation reporting
------------------------------
Every command execution is logged as a CalculationStep, capturing the exact
local transform matrix, the previous and resulting world transform matrices,
and the extracted state. MatrixCalculationReporter renders this log as a
full, numeric, step-by-step report (every matrix shown explicitly) suitable
for instructor review, saved to a text file via --calculations-file
(defaults to matrix_operations.txt) and optionally printed to console
via --show-calculations.

Run this file directly to be prompted for a start and end point, or with
--run-tests to execute the full automated test suite.
"""

from __future__ import annotations

import argparse
import math
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

try:
    import matplotlib.pyplot as plt

    _has_matplotlib = True
except ImportError:  # pragma: no cover - environment-dependent
    _has_matplotlib = False
    plt = None  # type: ignore[assignment]

MATPLOTLIB_AVAILABLE: bool = _has_matplotlib


# ======================================================================
# Constants
# ======================================================================
NUMERICAL_TOLERANCE: float = 1e-6
TRAJECTORY_LINE_COLOR: str = "#4B2C6F"   # Habib University / OSL purple
HEADING_ARROW_COLOR: str = "#C9A24B"     # Habib University / OSL gold
START_MARKER_COLOR: str = "#2E7D32"
END_MARKER_COLOR: str = "#B00020"

# A fixed mission used by the automated test suite (chained-transformation
# verification) and available via explicit command mode for scripting; the
# default CLI entry point no longer runs this automatically.
DEMO_MISSION_COMMANDS: List[str] = [
    "FORWARD 5",
    "ROTATE 90",
    "FORWARD 3",
    "ROTATE -45",
    "FORWARD 4",
    "ROTATE 135",
    "FORWARD 2",
    "BACKWARD 1",
]


# ======================================================================
# Command representation
# ======================================================================
class CommandType(Enum):
    """Supported rover movement command categories."""

    FORWARD = auto()
    BACKWARD = auto()
    ROTATE = auto()


@dataclass(frozen=True)
class Command:
    """An immutable, validated rover instruction.

    Attributes:
        command_type: The category of movement this command represents.
        value: Distance in world units for FORWARD/BACKWARD, or signed
            rotation angle in degrees for ROTATE (positive = counter-clockwise).
    """

    command_type: CommandType
    value: float


# ======================================================================
# Command parsing
# ======================================================================
class CommandParseError(ValueError):
    """Raised when a raw command string cannot be parsed into a Command."""


class CommandParser:
    """Converts raw command strings into structured Command objects."""

    _KEYWORD_TO_TYPE = {
        "FORWARD": CommandType.FORWARD,
        "FWD": CommandType.FORWARD,
        "BACKWARD": CommandType.BACKWARD,
        "BACK": CommandType.BACKWARD,
        "BWD": CommandType.BACKWARD,
        "ROTATE": CommandType.ROTATE,
        "ROT": CommandType.ROTATE,
        "TURN": CommandType.ROTATE,
    }

    @classmethod
    def parse_line(cls, raw_command: str) -> Command:
        """Parse a single '<ACTION> <VALUE>' command line.

        Raises:
            CommandParseError: If the line is empty, malformed, uses an
                unknown keyword, has a non-numeric value, or specifies a
                negative movement distance.
        """
        if not raw_command or not raw_command.strip():
            raise CommandParseError("Empty command string cannot be parsed.")

        tokens = raw_command.strip().split()
        if len(tokens) != 2:
            raise CommandParseError(
                f"Command '{raw_command.strip()}' must have exactly two "
                f"tokens: '<ACTION> <VALUE>'."
            )

        keyword, value_token = tokens[0].upper(), tokens[1]
        if keyword not in cls._KEYWORD_TO_TYPE:
            valid_keywords = ", ".join(sorted(cls._KEYWORD_TO_TYPE))
            raise CommandParseError(
                f"Unknown command keyword '{keyword}'. "
                f"Valid keywords are: {valid_keywords}."
            )

        try:
            value = float(value_token)
        except ValueError as exc:
            raise CommandParseError(
                f"Command value '{value_token}' is not a valid number."
            ) from exc

        if not math.isfinite(value):
            raise CommandParseError(
                f"Command value '{value_token}' must be finite."
            )

        command_type = cls._KEYWORD_TO_TYPE[keyword]
        if command_type in (CommandType.FORWARD, CommandType.BACKWARD) and value < 0:
            raise CommandParseError(
                "Movement distance must be non-negative. "
                "Use BACKWARD for reverse movement instead of a negative value."
            )

        return Command(command_type=command_type, value=value)

    @classmethod
    def parse_program(cls, raw_commands: Sequence[str]) -> List[Command]:
        """Parse an ordered sequence of raw command lines, skipping blanks
        and lines beginning with '#' (comments)."""
        parsed: List[Command] = []
        for line_number, raw_line in enumerate(raw_commands, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                parsed.append(cls.parse_line(stripped))
            except CommandParseError as exc:
                raise CommandParseError(f"Line {line_number}: {exc}") from exc
        return parsed


# ======================================================================
# Rover state
# ======================================================================
@dataclass
class RoverState:
    """A single snapshot of the rover's position and heading.

    Heading is stored internally in radians to keep all trigonometric
    computation in one unit system and avoid degree/radian mixing errors.
    """

    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0

    @property
    def heading_deg(self) -> float:
        """Heading in degrees, normalized to the range [-180, 180)."""
        degrees = math.degrees(self.heading_rad)
        return ((degrees + 180.0) % 360.0) - 180.0

    def as_homogeneous_vector(self) -> np.ndarray:
        """Return this state's position as a homogeneous coordinate vector."""
        return np.array([self.x, self.y, 1.0], dtype=np.float64)

    def copy(self) -> "RoverState":
        return RoverState(self.x, self.y, self.heading_rad)


# ======================================================================
# Matrix Operations Engine
# ======================================================================
class TransformationEngine:
    """Builds and composes homogeneous transformation matrices.

    All matrices are 3x3 homogeneous transforms operating on
    [x, y, 1]^T column vectors.
    """

    @staticmethod
    def rotation_matrix(angle_rad: float) -> np.ndarray:
        """Pure rotation about the local origin by `angle_rad` radians."""
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        return np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def translation_matrix(delta_x: float, delta_y: float) -> np.ndarray:
        """Pure translation by (delta_x, delta_y)."""
        return np.array(
            [
                [1.0, 0.0, delta_x],
                [0.0, 1.0, delta_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def combined_matrix(angle_rad: float, delta_x: float, delta_y: float) -> np.ndarray:
        """Combined rotation-then-translation homogeneous transform.

        Equivalent to translation_matrix(delta_x, delta_y) @ rotation_matrix(angle_rad),
        computed directly for efficiency.
        """
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        return np.array(
            [
                [cos_a, -sin_a, delta_x],
                [sin_a, cos_a, delta_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def local_transform_for_command(command: Command) -> np.ndarray:
        """Build the local-frame transformation matrix for a single command."""
        if command.command_type is CommandType.FORWARD:
            return TransformationEngine.translation_matrix(command.value, 0.0)
        if command.command_type is CommandType.BACKWARD:
            return TransformationEngine.translation_matrix(-command.value, 0.0)
        if command.command_type is CommandType.ROTATE:
            return TransformationEngine.rotation_matrix(math.radians(command.value))
        raise ValueError(f"Unsupported command type: {command.command_type!r}")

    @staticmethod
    def compose_local_transforms(
        commands: Sequence[Command],
        initial_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compose an ordered sequence of local-frame command transforms into
        a single composite matrix via chained matrix multiplication.

        composite = W_0 @ T_1 @ T_2 @ ... @ T_n, where W_0 is the rover's
        starting world transform (identity if the rover starts at the
        origin with zero heading, which is the default). Because each T_i
        is expressed in the rover's local frame at the moment it is issued,
        right-multiplication correctly accumulates world-frame pose
        regardless of where the rover started.
        """
        composite = (
            np.identity(3, dtype=np.float64)
            if initial_transform is None
            else initial_transform.copy()
        )
        for command in commands:
            composite = composite @ TransformationEngine.local_transform_for_command(command)
        return composite

    @staticmethod
    def extract_state_from_world_transform(matrix: np.ndarray) -> RoverState:
        """Recover (x, y, heading) from a 3x3 homogeneous world transform."""
        x, y = float(matrix[0, 2]), float(matrix[1, 2])
        heading_rad = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        return RoverState(x=x, y=y, heading_rad=heading_rad)


# ======================================================================
# Calculation logging (for instructor / professor review)
# ======================================================================
@dataclass(frozen=True)
class CalculationStep:
    """A complete, numeric record of one command's matrix arithmetic.

    Captures every matrix involved in updating the rover's pose for a
    single command, so the full computation W_n = W_(n-1) @ T_local can be
    displayed and verified independently of the running program.
    """

    step_number: int
    command: Command
    local_transform: np.ndarray
    previous_world_transform: np.ndarray
    new_world_transform: np.ndarray
    resulting_state: RoverState


# ======================================================================
# Rover: state machine + trajectory storage
# ======================================================================
class Rover:
    """A simulated rover that tracks pose and records its full trajectory."""

    def __init__(
        self,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_heading_deg: float = 0.0,
    ) -> None:
        self.reset(initial_x, initial_y, initial_heading_deg)

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading_deg: float = 0.0,
    ) -> None:
        """Reset the rover to a given pose and clear its trajectory history."""
        self._validate_finite(x, "x")
        self._validate_finite(y, "y")
        self._validate_finite(heading_deg, "heading_deg")

        heading_rad = math.radians(heading_deg)
        self._world_transform: np.ndarray = TransformationEngine.combined_matrix(
            heading_rad, x, y
        )
        self.state: RoverState = RoverState(x=x, y=y, heading_rad=heading_rad)
        self.trajectory: List[RoverState] = [self.state.copy()]
        self.calculation_log: List[CalculationStep] = []

    @staticmethod
    def _validate_finite(value: float, name: str) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(value):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"'{name}' must be a finite number, got {value!r}.")

    def apply_command(self, command: Command) -> RoverState:
        """Apply a single command, update world pose, and record both the
        trajectory and the full matrix arithmetic behind the update."""
        if not isinstance(command, Command):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected a Command instance, got {type(command).__name__}.")

        previous_world_transform = self._world_transform.copy()
        local_transform = TransformationEngine.local_transform_for_command(command)
        new_world_transform = previous_world_transform @ local_transform

        self._world_transform = new_world_transform
        self.state = TransformationEngine.extract_state_from_world_transform(
            new_world_transform
        )
        self.trajectory.append(self.state.copy())
        self.calculation_log.append(
            CalculationStep(
                step_number=len(self.calculation_log) + 1,
                command=command,
                local_transform=local_transform.copy(),
                previous_world_transform=previous_world_transform,
                new_world_transform=new_world_transform.copy(),
                resulting_state=self.state.copy(),
            )
        )
        return self.state

    def run_program(self, commands: Sequence[Command]) -> RoverState:
        """Apply an ordered sequence of commands, returning the final state."""
        for command in commands:
            self.apply_command(command)
        return self.state

    def get_trajectory_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the recorded trajectory as (xs, ys, heading_radians) arrays."""
        xs = np.array([s.x for s in self.trajectory], dtype=np.float64)
        ys = np.array([s.y for s in self.trajectory], dtype=np.float64)
        thetas = np.array([s.heading_rad for s in self.trajectory], dtype=np.float64)
        return xs, ys, thetas


# ======================================================================
# Visualization
# ======================================================================
class TrajectoryVisualizer:
    """Renders a rover's recorded trajectory using Matplotlib."""

    @staticmethod
    def plot(
        rover: Rover,
        title: str = "Rover Navigation Simulator - Trajectory",
        save_path: Optional[str] = None,
        show: bool = True,
        arrow_stride: int = 1,
    ) -> None:
        """Plot the rover's path with start/end markers and heading arrows.

        Raises:
            RuntimeError: If Matplotlib is not installed.
            ValueError: If the rover has no recorded trajectory.
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            raise RuntimeError(
                "Matplotlib is required for visualization but is not installed. "
                "Install it with: pip install matplotlib"
            )
        assert plt is not None
        if len(rover.trajectory) == 0:
            raise ValueError("Rover has no recorded trajectory to plot.")

        xs, ys, thetas = rover.get_trajectory_arrays()

        figure, axes = plt.subplots(figsize=(8, 8))  # pyright: ignore[reportUnknownMemberType, reportPossiblyUnboundVariable]
        axes.plot(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            xs, ys,
            color=TRAJECTORY_LINE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Trajectory",
            zorder=2,
        )
        axes.scatter(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            [xs[0]], [ys[0]], color=START_MARKER_COLOR, s=90, zorder=5, label="Start"
        )
        axes.scatter(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            [xs[-1]], [ys[-1]], color=END_MARKER_COLOR, s=90, zorder=5, label="End"
        )

        arrow_length = TrajectoryVisualizer._compute_arrow_length(xs, ys)
        stride = max(1, arrow_stride)
        for index in range(0, len(xs), stride):
            dx = arrow_length * math.cos(thetas[index])
            dy = arrow_length * math.sin(thetas[index])
            axes.annotate(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                "",
                xy=(xs[index] + dx, ys[index] + dy),
                xytext=(xs[index], ys[index]),
                arrowprops=dict(arrowstyle="->", color=HEADING_ARROW_COLOR, lw=1.5),
                zorder=4,
            )

        axes.set_title(title, fontsize=14, fontweight="bold")  # pyright: ignore[reportUnknownMemberType]
        axes.set_xlabel("X Position")  # pyright: ignore[reportUnknownMemberType]
        axes.set_ylabel("Y Position")  # pyright: ignore[reportUnknownMemberType]
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, linestyle="--", alpha=0.4)  # pyright: ignore[reportUnknownMemberType]
        axes.legend(loc="best")  # pyright: ignore[reportUnknownMemberType]

        if save_path:
            figure.savefig(save_path, dpi=200, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        if show:
            plt.show()  # pyright: ignore[reportUnknownMemberType, reportPossiblyUnboundVariable]
        plt.close(figure)  # pyright: ignore[reportPossiblyUnboundVariable]

    @staticmethod
    def _compute_arrow_length(xs: np.ndarray, ys: np.ndarray) -> float:
        """Scale heading-arrow length relative to the trajectory's extent."""
        span_x = float(xs.max() - xs.min()) if len(xs) > 1 else 1.0
        span_y = float(ys.max() - ys.min()) if len(ys) > 1 else 1.0
        span = max(span_x, span_y, 1.0)
        return span * 0.06


def print_trajectory_table(rover: Rover) -> None:
    """Print the rover's recorded trajectory as a plain-text table."""
    header = f"{'Step':>4} | {'X':>12} | {'Y':>12} | {'Heading (deg)':>14}"
    print(header)
    print("-" * len(header))
    for step, state in enumerate(rover.trajectory):
        print(
            f"{step:>4} | {state.x:>12.4f} | {state.y:>12.4f} | "
            f"{state.heading_deg:>14.4f}"
        )


def format_matrix(matrix: np.ndarray, decimals: int = 4) -> str:
    """Format a 3x3 matrix as aligned, bracketed rows for review, e.g.:

        [   1.0000   0.0000   5.0000 ]
        [   0.0000   1.0000   0.0000 ]
        [   0.0000   0.0000   1.0000 ]
    """
    rows: List[str] = []
    for row in matrix:
        formatted_values = [f"{value:>9.{decimals}f}" for value in row]
        rows.append("[ " + "  ".join(formatted_values) + " ]")
    return "\n".join(rows)


def describe_command(command: Command) -> str:
    """Human-readable description of a command for the calculation report."""
    if command.command_type is CommandType.FORWARD:
        return f"FORWARD {command.value:g} units"
    if command.command_type is CommandType.BACKWARD:
        return f"BACKWARD {command.value:g} units"
    if command.command_type is CommandType.ROTATE:
        return f"ROTATE {command.value:g} degrees"
    raise ValueError(f"Unsupported command type: {command.command_type!r}")


class MatrixCalculationReporter:
    """Renders a rover's recorded calculation log as a full, numeric,
    step-by-step report: every matrix built and every multiplication
    performed, suitable for direct instructor review."""

    @staticmethod
    def generate_report(rover: Rover) -> str:
        """Build the complete calculation report as a single string."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("ROVER NAVIGATION SIMULATOR - MATRIX CALCULATION REPORT")
        lines.append("=" * 70)

        initial_state = rover.trajectory[0]
        initial_world_transform = TransformationEngine.combined_matrix(
            initial_state.heading_rad, initial_state.x, initial_state.y
        )
        lines.append("")
        lines.append(
            f"Initial state:  x = {initial_state.x:.4f}   "
            f"y = {initial_state.y:.4f}   "
            f"heading = {initial_state.heading_deg:.4f} deg ({initial_state.heading_rad:.4f} rad)"
        )
        lines.append("")
        lines.append("Initial world transform matrix  W_0:")
        lines.append(format_matrix(initial_world_transform))

        for step in rover.calculation_log:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"STEP {step.step_number}: {describe_command(step.command)}")
            lines.append("-" * 70)

            cmd = step.command
            lines.append("")
            if cmd.command_type in (CommandType.FORWARD, CommandType.BACKWARD):
                dist = cmd.value if cmd.command_type is CommandType.FORWARD else -cmd.value
                lines.append(f"Command Parameter: Translation distance d = {dist:.4f} units")
                lines.append("Local Matrix Formula: T_local = [[1, 0, d], [0, 1, 0], [0, 0, 1]]")
            elif cmd.command_type is CommandType.ROTATE:
                angle_deg = cmd.value
                angle_rad = math.radians(angle_deg)
                lines.append(f"Command Parameter: Rotation angle theta = {angle_deg:.4f} deg ({angle_rad:.4f} rad)")
                lines.append("Local Matrix Formula: T_local = [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]")
                lines.append(f"  cos({angle_deg:.4f} deg) = {math.cos(angle_rad):.4f}")
                lines.append(f"  sin({angle_deg:.4f} deg) = {math.sin(angle_rad):.4f}")

            lines.append("")
            lines.append("Local-frame transformation matrix  T_local:")
            lines.append(format_matrix(step.local_transform))

            lines.append("")
            lines.append("Previous world transform  W_(n-1):")
            lines.append(format_matrix(step.previous_world_transform))

            lines.append("")
            lines.append("Matrix multiplication performed:  W_n = W_(n-1) . T_local")
            lines.append("Element-by-element dot product breakdown (W_n[i,j] = sum_k W_(n-1)[i,k] * T_local[k,j]):")
            W_prev = step.previous_world_transform
            T_loc = step.local_transform
            W_new = step.new_world_transform
            for i in range(3):
                row_parts = []
                for j in range(3):
                    terms = " + ".join([f"({W_prev[i,k]:.4f} * {T_loc[k,j]:.4f})" for k in range(3)])
                    row_parts.append(f"  W_n[{i},{j}] = {terms} = {W_new[i,j]:.4f}")
                lines.append("\n".join(row_parts))

            lines.append("")
            lines.append("Resulting world transform  W_n:")
            lines.append(format_matrix(step.new_world_transform))

            lines.append("")
            heading_rad = math.atan2(float(W_new[1, 0]), float(W_new[0, 0]))
            heading_deg = TrajectoryPlanner.normalize_degrees(math.degrees(heading_rad))
            lines.append("State Extraction Formulas from Homogeneous World Matrix W_n:")
            lines.append(f"  Position X = W_n[0, 2] = {W_new[0, 2]:.4f}")
            lines.append(f"  Position Y = W_n[1, 2] = {W_new[1, 2]:.4f}")
            lines.append(f"  Heading    = atan2(W_n[1,0], W_n[0,0]) = atan2({W_new[1,0]:.4f}, {W_new[0,0]:.4f}) = {heading_rad:.4f} rad ({heading_deg:.4f} deg)")

            lines.append("")
            lines.append(
                f"Extracted state:  x = {step.resulting_state.x:.4f}   "
                f"y = {step.resulting_state.y:.4f}   "
                f"heading = {step.resulting_state.heading_deg:.4f} deg"
            )

        lines.append("")
        lines.append("=" * 70)
        final_state = rover.state
        lines.append(
            f"FINAL STATE:  x = {final_state.x:.4f}   "
            f"y = {final_state.y:.4f}   "
            f"heading = {final_state.heading_deg:.4f} deg"
        )
        lines.append("=" * 70)

        commands = [step.command for step in rover.calculation_log]
        if commands:
            composite_matrix = TransformationEngine.compose_local_transforms(
                commands, initial_transform=initial_world_transform
            )
            composite_state = TransformationEngine.extract_state_from_world_transform(
                composite_matrix
            )
            position_matches = (
                abs(composite_state.x - final_state.x) < NUMERICAL_TOLERANCE
                and abs(composite_state.y - final_state.y) < NUMERICAL_TOLERANCE
            )
            lines.append("")
            lines.append(
                "VERIFICATION: single composite matrix (all local transforms "
                "multiplied together, T_1 . T_2 . ... . T_n) versus the "
                "step-by-step result above:"
            )
            lines.append("")
            lines.append("Composite matrix:")
            lines.append(format_matrix(composite_matrix))
            lines.append("")
            lines.append(
                f"Composite-derived state:  x = {composite_state.x:.4f}   "
                f"y = {composite_state.y:.4f}   "
                f"heading = {composite_state.heading_deg:.4f} deg"
            )
            lines.append(
                f"Matches step-by-step final state: "
                f"{'YES' if position_matches else 'NO'}"
            )

        return "\n".join(lines)

    @staticmethod
    def save_report(rover: Rover, path: str, preamble: Optional[str] = None) -> None:
        """Write the full calculation report to a text file, optionally
        prefixed with additional context (e.g. a route derivation).

        Raises:
            OSError: If the file cannot be written.
        """
        report_text = MatrixCalculationReporter.generate_report(rover)
        full_text = f"{preamble}\n\n{report_text}" if preamble else report_text
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(full_text)
                handle.write("\n")
        except OSError as exc:
            raise OSError(f"Could not write calculation report to '{path}': {exc}") from exc


# ======================================================================
# Trajectory planning: derive a route from a start point to an end point
# ======================================================================
@dataclass(frozen=True)
class RouteDerivation:
    """A complete, numeric record of how a rotate-then-drive route was
    derived from a start pose and a target end point.

    The route uses the simplest motion that this rover's command set can
    express: rotate to face the target, drive straight to it, then
    (optionally) rotate again to face a requested final heading.
    """

    start_x: float
    start_y: float
    start_heading_deg: float
    end_x: float
    end_y: float
    end_heading_deg: Optional[float]
    delta_x: float
    delta_y: float
    distance: float
    heading_to_target_deg: float
    initial_turn_deg: float
    final_turn_deg: Optional[float]
    commands: List[Command]


class TrajectoryPlanner:
    """Derives the command sequence that takes the rover from a start pose
    to a target end point, directly from the geometry of the displacement
    vector between them."""

    @staticmethod
    def plan_route_to_point(
        start_x: float,
        start_y: float,
        start_heading_deg: float,
        end_x: float,
        end_y: float,
        end_heading_deg: Optional[float] = None,
    ) -> RouteDerivation:
        """Compute the rotate-then-drive (optionally plus a final rotate)
        command sequence connecting the start pose to the end point.

        Raises:
            ValueError: If any input is not a finite number.
        """
        for name, value in (
            ("start_x", start_x),
            ("start_y", start_y),
            ("start_heading_deg", start_heading_deg),
            ("end_x", end_x),
            ("end_y", end_y),
        ):
            if not math.isfinite(value):
                raise ValueError(f"'{name}' must be a finite number, got {value!r}.")
        if end_heading_deg is not None and not math.isfinite(end_heading_deg):
            raise ValueError(
                f"'end_heading_deg' must be a finite number, got {end_heading_deg!r}."
            )

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        distance = math.hypot(delta_x, delta_y)

        commands: List[Command] = []

        if distance > NUMERICAL_TOLERANCE:
            heading_to_target_deg = math.degrees(math.atan2(delta_y, delta_x))
            initial_turn_deg = TrajectoryPlanner.normalize_degrees(
                heading_to_target_deg - start_heading_deg
            )
            if abs(initial_turn_deg) > NUMERICAL_TOLERANCE:
                commands.append(Command(CommandType.ROTATE, initial_turn_deg))
            commands.append(Command(CommandType.FORWARD, distance))
            heading_after_drive = heading_to_target_deg
        else:
            heading_to_target_deg = start_heading_deg
            initial_turn_deg = 0.0
            heading_after_drive = start_heading_deg

        final_turn_deg: Optional[float] = None
        if end_heading_deg is not None:
            final_turn_deg = TrajectoryPlanner.normalize_degrees(
                end_heading_deg - heading_after_drive
            )
            if abs(final_turn_deg) > NUMERICAL_TOLERANCE:
                commands.append(Command(CommandType.ROTATE, final_turn_deg))

        return RouteDerivation(
            start_x=start_x,
            start_y=start_y,
            start_heading_deg=start_heading_deg,
            end_x=end_x,
            end_y=end_y,
            end_heading_deg=end_heading_deg,
            delta_x=delta_x,
            delta_y=delta_y,
            distance=distance,
            heading_to_target_deg=heading_to_target_deg,
            initial_turn_deg=initial_turn_deg,
            final_turn_deg=final_turn_deg,
            commands=commands,
        )

    @staticmethod
    def normalize_degrees(angle_deg: float) -> float:
        """Normalize an angle to the range [-180, 180) degrees."""
        return ((angle_deg + 180.0) % 360.0) - 180.0


def format_route_derivation(derivation: RouteDerivation) -> str:
    """Render a RouteDerivation as a step-by-step geometric derivation,
    showing exactly how the rotate/forward commands were computed from the
    start and end points."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("ROUTE PLANNING DERIVATION")
    lines.append("=" * 70)

    lines.append("")
    lines.append(
        f"Start point:  ({derivation.start_x:.4f}, {derivation.start_y:.4f})   "
        f"heading = {derivation.start_heading_deg:.4f} deg"
    )
    end_heading_text = (
        f"{derivation.end_heading_deg:.4f} deg"
        if derivation.end_heading_deg is not None
        else "not specified"
    )
    lines.append(
        f"End point:    ({derivation.end_x:.4f}, {derivation.end_y:.4f})   "
        f"heading = {end_heading_text}"
    )

    lines.append("")
    lines.append("Displacement vector:")
    lines.append(
        f"  delta_x = end_x - start_x = {derivation.end_x:.4f} - "
        f"{derivation.start_x:.4f} = {derivation.delta_x:.4f}"
    )
    lines.append(
        f"  delta_y = end_y - start_y = {derivation.end_y:.4f} - "
        f"{derivation.start_y:.4f} = {derivation.delta_y:.4f}"
    )

    lines.append("")
    lines.append("Distance to target:")
    lines.append(
        f"  distance = sqrt(delta_x^2 + delta_y^2) = "
        f"sqrt({derivation.delta_x:.4f}^2 + {derivation.delta_y:.4f}^2) = "
        f"{derivation.distance:.4f}"
    )

    lines.append("")
    lines.append("Heading required to face the target:")
    lines.append(
        f"  heading_to_target = atan2(delta_y, delta_x) = "
        f"atan2({derivation.delta_y:.4f}, {derivation.delta_x:.4f}) = "
        f"{derivation.heading_to_target_deg:.4f} deg"
    )

    lines.append("")
    lines.append("Initial rotation needed:")
    lines.append(
        f"  initial_turn = heading_to_target - start_heading = "
        f"{derivation.heading_to_target_deg:.4f} - {derivation.start_heading_deg:.4f} "
        f"= {derivation.initial_turn_deg:.4f} deg (normalized to [-180, 180))"
    )

    if derivation.final_turn_deg is not None:
        lines.append("")
        lines.append("Final rotation needed to reach the requested end heading:")
        lines.append(
            f"  final_turn = end_heading - heading_to_target = "
            f"{derivation.end_heading_deg:.4f} - {derivation.heading_to_target_deg:.4f} "
            f"= {derivation.final_turn_deg:.4f} deg (normalized to [-180, 180))"
        )

    lines.append("")
    lines.append("Derived command sequence:")
    if not derivation.commands:
        lines.append("  (start and end points already coincide; no movement required)")
    else:
        for index, command in enumerate(derivation.commands, start=1):
            lines.append(f"  {index}. {describe_command(command)}")

    return "\n".join(lines)


# ======================================================================
# Interactive prompts
# ======================================================================
def compass_direction_name(heading_deg: float) -> str:
    """Return a cardinal or intercardinal direction label for a heading in degrees."""
    deg = heading_deg % 360.0
    if deg < 0:
        deg += 360.0
    directions = [
        "East (+X)",
        "North-East",
        "North (+Y)",
        "North-West",
        "West (-X)",
        "South-West",
        "South (-Y)",
        "South-East",
    ]
    index = int((deg + 22.5) // 45) % 8
    return directions[index]


def prompt_float(
    prompt_text: str,
    default: Optional[float] = None,
    help_text: Optional[str] = None,
    suggested_example: Optional[str] = None,
) -> float:
    """Prompt the user for a required float value, re-prompting on invalid
    input. If `default` is given, pressing Enter with no input accepts it."""
    suffix_parts: List[str] = []
    if default is not None:
        suffix_parts.append(f"default: {default:g}")
    if suggested_example and default is None:
        suffix_parts.append(f"e.g., {suggested_example}")

    suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""

    if help_text:
        print(f"  [i] {help_text}")

    while True:
        raw_input_text = input(f"  -> {prompt_text}{suffix}: ").strip()
        if not raw_input_text and default is not None:
            return default
        try:
            value = float(raw_input_text)
        except ValueError:
            print("  [x] Invalid input: Please enter a valid number (e.g., 0.0, 10.5, -5.0).")
            continue
        if not math.isfinite(value):
            print("  [x] Invalid input: Value must be finite.")
            continue
        return value


def prompt_optional_float(
    prompt_text: str,
    help_text: Optional[str] = None,
    suggested_example: Optional[str] = None,
) -> Optional[float]:
    """Prompt for an optional float; pressing Enter with no input skips it."""
    suffix_parts = ["leave blank to skip"]
    if suggested_example:
        suffix_parts.append(f"e.g., {suggested_example}")

    suffix = f" [{', '.join(suffix_parts)}]"

    if help_text:
        print(f"  [i] {help_text}")

    while True:
        raw_input_text = input(f"  -> {prompt_text}{suffix}: ").strip()
        if not raw_input_text:
            return None
        try:
            value = float(raw_input_text)
        except ValueError:
            print("  [x] Invalid input: Please enter a valid number, or leave blank to skip.")
            continue
        if not math.isfinite(value):
            print("  [x] Invalid input: Value must be finite.")
            continue
        return value


def prompt_start_and_end_points() -> Tuple[float, float, float, float, float, Optional[float]]:
    """Interactively collect the rover's start pose and target end point.

    Returns:
        (start_x, start_y, start_heading_deg, end_x, end_y, end_heading_deg)
        where end_heading_deg may be None if the user leaves it blank.
    """
    print("=" * 70)
    print("        ROVER NAVIGATION SIMULATOR - INTERACTIVE POSE SETUP")
    print("=" * 70)
    print(" Enter the rover's starting pose and target destination coordinates.")
    print()
    print(" COORDINATE REFERENCE GUIDE:")
    print("   * X Axis       : Horizontal position in meters (+ East / - West)")
    print("   * Y Axis       : Vertical position in meters (+ North / - South)")
    print("   * Heading (deg): Orientation (0 deg = East, 90 deg = North, 180 deg = West, 270 deg = South)")
    print("   * Shortcuts    : Press [Enter] to accept default or suggested values.")
    print("=" * 70)
    print()

    print("----------------------------------------------------------------------")
    print(" [1/2] ROVER STARTING POSE (Initial Position & Orientation)")
    print("----------------------------------------------------------------------")
    start_x = prompt_float(
        "Start X",
        default=0.0,
        help_text="Horizontal position in meters (0.0 = Origin).",
        suggested_example="0.0",
    )
    start_y = prompt_float(
        "Start Y",
        default=0.0,
        help_text="Vertical position in meters (0.0 = Origin).",
        suggested_example="0.0",
    )
    start_heading_deg = prompt_float(
        "Start heading (degrees)",
        default=0.0,
        help_text="Initial orientation (0 deg = facing East, 90 deg = facing North).",
        suggested_example="0.0",
    )

    print()
    print("----------------------------------------------------------------------")
    print(" [2/2] TARGET DESTINATION POSE (Goal Position & Orientation)")
    print("----------------------------------------------------------------------")
    end_x = prompt_float(
        "End X",
        help_text="Target horizontal position in meters.",
        suggested_example="10.0",
    )
    end_y = prompt_float(
        "End Y",
        help_text="Target vertical position in meters.",
        suggested_example="10.0",
    )
    end_heading_deg = prompt_optional_float(
        "End heading (degrees)",
        help_text="Optional orientation at destination (leave blank to keep arrival direction).",
        suggested_example="90.0",
    )

    dx = end_x - start_x
    dy = end_y - start_y
    dist = math.hypot(dx, dy)
    bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0

    start_dir = compass_direction_name(start_heading_deg)
    end_dir_str = (
        f"{end_heading_deg:g} deg ({compass_direction_name(end_heading_deg)})"
        if end_heading_deg is not None
        else "Auto (Arrival bearing)"
    )
    bearing_dir = compass_direction_name(bearing_deg)

    print()
    print("======================================================================")
    print("               MISSION POSE CONFIGURATION SUMMARY")
    print("======================================================================")
    print(f"  * Starting Pose : X = {start_x:g} m, Y = {start_y:g} m | Heading = {start_heading_deg:g} deg ({start_dir})")
    print(f"  * Target Pose   : X = {end_x:g} m, Y = {end_y:g} m | Heading = {end_dir_str}")
    print(f"  * Trajectory    : Direct Distance = {dist:.2f} m | Bearing = {bearing_deg:.1f} deg ({bearing_dir})")
    print("======================================================================")
    print()

    return start_x, start_y, start_heading_deg, end_x, end_y, end_heading_deg


# ======================================================================
# Automated tests (Straight / Rotation / Combined / Chained / Boundary)
# ======================================================================
class TestStraightMovement(unittest.TestCase):
    """Forward/backward commands should translate position without
    changing heading."""

    def test_forward_movement_along_zero_heading(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.FORWARD, 5.0))
        self.assertAlmostEqual(rover.state.x, 5.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.y, 0.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.heading_rad, 0.0, delta=NUMERICAL_TOLERANCE)

    def test_backward_movement(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.BACKWARD, 3.0))
        self.assertAlmostEqual(rover.state.x, -3.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.y, 0.0, delta=NUMERICAL_TOLERANCE)


class TestRotationOnly(unittest.TestCase):
    """Rotate commands should change heading without changing position."""

    def test_rotation_does_not_change_position(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.ROTATE, 90.0))
        self.assertAlmostEqual(rover.state.x, 0.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.y, 0.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.heading_deg, 90.0, delta=1e-4)

    def test_full_rotation_returns_to_original_heading(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.ROTATE, 360.0))
        self.assertAlmostEqual(math.sin(rover.state.heading_rad), 0.0, delta=1e-9)
        self.assertAlmostEqual(math.cos(rover.state.heading_rad), 1.0, delta=1e-9)


class TestCombinedRotationTranslation(unittest.TestCase):
    """A rotation followed by a translation should move along the new heading."""

    def test_rotate_then_move_travels_along_new_heading(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.ROTATE, 90.0))
        rover.apply_command(Command(CommandType.FORWARD, 4.0))
        self.assertAlmostEqual(rover.state.x, 0.0, delta=1e-4)
        self.assertAlmostEqual(rover.state.y, 4.0, delta=1e-4)


class TestChainedTransformations(unittest.TestCase):
    """A composite matrix built from chained transforms must match the
    result of applying each command sequentially."""

    def test_composite_matrix_matches_sequential_application(self) -> None:
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)

        sequential_rover = Rover()
        sequential_rover.run_program(commands)

        composite_matrix = TransformationEngine.compose_local_transforms(commands)
        composite_state = TransformationEngine.extract_state_from_world_transform(
            composite_matrix
        )

        self.assertAlmostEqual(
            composite_state.x, sequential_rover.state.x, delta=NUMERICAL_TOLERANCE
        )
        self.assertAlmostEqual(
            composite_state.y, sequential_rover.state.y, delta=NUMERICAL_TOLERANCE
        )
        self.assertAlmostEqual(
            composite_state.heading_rad,
            sequential_rover.state.heading_rad,
            delta=NUMERICAL_TOLERANCE,
        )

    def test_composite_matrix_matches_with_nonzero_starting_pose(self) -> None:
        """Regression test: the composite must account for a non-default
        initial pose, not silently assume the rover starts at the origin."""
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)

        sequential_rover = Rover(initial_x=2.0, initial_y=1.0, initial_heading_deg=30.0)
        sequential_rover.run_program(commands)

        initial_world_transform = TransformationEngine.combined_matrix(
            math.radians(30.0), 2.0, 1.0
        )
        composite_matrix = TransformationEngine.compose_local_transforms(
            commands, initial_transform=initial_world_transform
        )
        composite_state = TransformationEngine.extract_state_from_world_transform(
            composite_matrix
        )

        self.assertAlmostEqual(
            composite_state.x, sequential_rover.state.x, delta=NUMERICAL_TOLERANCE
        )
        self.assertAlmostEqual(
            composite_state.y, sequential_rover.state.y, delta=NUMERICAL_TOLERANCE
        )
        self.assertAlmostEqual(
            composite_state.heading_rad,
            sequential_rover.state.heading_rad,
            delta=NUMERICAL_TOLERANCE,
        )


class TestBoundaryCases(unittest.TestCase):
    """Zero-value, full-rotation, large-value, and invalid inputs."""

    def test_zero_distance_movement(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.FORWARD, 0.0))
        self.assertAlmostEqual(rover.state.x, 0.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.y, 0.0, delta=NUMERICAL_TOLERANCE)

    def test_zero_degree_rotation(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.ROTATE, 0.0))
        self.assertAlmostEqual(rover.state.heading_rad, 0.0, delta=NUMERICAL_TOLERANCE)

    def test_large_distance_numerical_stability(self) -> None:
        rover = Rover()
        rover.apply_command(Command(CommandType.FORWARD, 1.0e6))
        self.assertTrue(math.isfinite(rover.state.x))
        self.assertAlmostEqual(rover.state.x, 1.0e6, delta=1.0)

    def test_negative_movement_value_rejected(self) -> None:
        with self.assertRaises(CommandParseError):
            CommandParser.parse_line("FORWARD -5")

    def test_unknown_command_keyword_rejected(self) -> None:
        with self.assertRaises(CommandParseError):
            CommandParser.parse_line("FLY 10")

    def test_malformed_command_missing_value_rejected(self) -> None:
        with self.assertRaises(CommandParseError):
            CommandParser.parse_line("FORWARD")

    def test_non_numeric_value_rejected(self) -> None:
        with self.assertRaises(CommandParseError):
            CommandParser.parse_line("FORWARD abc")

    def test_empty_command_rejected(self) -> None:
        with self.assertRaises(CommandParseError):
            CommandParser.parse_line("")

    def test_invalid_reset_pose_rejected(self) -> None:
        rover = Rover()
        with self.assertRaises(ValueError):
            rover.reset(x=float("nan"), y=0.0, heading_deg=0.0)


class TestManualConfiguration(unittest.TestCase):
    """The rover's start pose and command program must be fully
    configurable at runtime rather than hardcoded."""

    def test_custom_initial_pose(self) -> None:
        rover = Rover(initial_x=10.0, initial_y=-5.0, initial_heading_deg=90.0)
        self.assertAlmostEqual(rover.state.x, 10.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.y, -5.0, delta=NUMERICAL_TOLERANCE)
        self.assertAlmostEqual(rover.state.heading_deg, 90.0, delta=1e-4)

    def test_custom_initial_pose_affects_final_end_point(self) -> None:
        rover_a = Rover(initial_x=0.0, initial_y=0.0)
        rover_b = Rover(initial_x=100.0, initial_y=50.0)
        command = Command(CommandType.FORWARD, 5.0)
        rover_a.apply_command(command)
        rover_b.apply_command(command)
        self.assertNotAlmostEqual(rover_a.state.x, rover_b.state.x)

    def test_inline_command_string_parsing(self) -> None:
        raw_commands = resolve_inline_commands("FORWARD 5;ROTATE 90;FORWARD 2")
        commands = CommandParser.parse_program(raw_commands)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0].command_type, CommandType.FORWARD)
        self.assertEqual(commands[1].command_type, CommandType.ROTATE)

    def test_inline_command_string_with_newlines(self) -> None:
        raw_commands = resolve_inline_commands("FORWARD 5\nROTATE 90\nFORWARD 2")
        commands = CommandParser.parse_program(raw_commands)
        self.assertEqual(len(commands), 3)


class TestMatrixCalculationLog(unittest.TestCase):
    """The rover must retain a complete, verifiable record of every matrix
    computation performed, for instructor review."""

    def test_calculation_log_length_matches_command_count(self) -> None:
        rover = Rover()
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)
        rover.run_program(commands)
        self.assertEqual(len(rover.calculation_log), len(commands))

    def test_calculation_log_multiplication_is_internally_consistent(self) -> None:
        rover = Rover()
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)
        rover.run_program(commands)
        for step in rover.calculation_log:
            recomputed = step.previous_world_transform @ step.local_transform
            self.assertTrue(
                np.allclose(recomputed, step.new_world_transform, atol=NUMERICAL_TOLERANCE)
            )

    def test_calculation_log_chains_between_steps(self) -> None:
        rover = Rover()
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)
        rover.run_program(commands)
        for previous_step, next_step in zip(rover.calculation_log, rover.calculation_log[1:]):
            self.assertTrue(
                np.allclose(
                    previous_step.new_world_transform,
                    next_step.previous_world_transform,
                    atol=NUMERICAL_TOLERANCE,
                )
            )

    def test_report_generation_contains_all_steps(self) -> None:
        rover = Rover()
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)
        rover.run_program(commands)
        report = MatrixCalculationReporter.generate_report(rover)
        self.assertIn("MATRIX CALCULATION REPORT", report)
        for step_number in range(1, len(commands) + 1):
            self.assertIn(f"STEP {step_number}:", report)
        self.assertIn("FINAL STATE", report)
        self.assertIn("Matches step-by-step final state: YES", report)

    def test_report_verification_passes_with_nonzero_starting_pose(self) -> None:
        rover = Rover(initial_x=2.0, initial_y=1.0, initial_heading_deg=30.0)
        commands = CommandParser.parse_program(DEMO_MISSION_COMMANDS)
        rover.run_program(commands)
        report = MatrixCalculationReporter.generate_report(rover)
        self.assertIn("Matches step-by-step final state: YES", report)

    def test_format_matrix_produces_three_rows(self) -> None:
        formatted = format_matrix(np.identity(3))
        self.assertEqual(len(formatted.splitlines()), 3)


class TestTrajectoryPlanner(unittest.TestCase):
    """The planner must derive a rotate-then-drive route from a start pose
    to a target end point, matching the geometry of the displacement."""

    def test_straight_line_requires_no_initial_rotation(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(0, 0, 0, 5, 0)
        self.assertEqual(len(derivation.commands), 1)
        self.assertIs(derivation.commands[0].command_type, CommandType.FORWARD)
        self.assertAlmostEqual(derivation.commands[0].value, 5.0, delta=NUMERICAL_TOLERANCE)

    def test_perpendicular_target_requires_90_degree_turn(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(0, 0, 0, 0, 5)
        self.assertEqual(len(derivation.commands), 2)
        self.assertIs(derivation.commands[0].command_type, CommandType.ROTATE)
        self.assertAlmostEqual(derivation.commands[0].value, 90.0, delta=1e-4)
        self.assertIs(derivation.commands[1].command_type, CommandType.FORWARD)
        self.assertAlmostEqual(derivation.commands[1].value, 5.0, delta=NUMERICAL_TOLERANCE)

    def test_requested_end_heading_appends_final_rotation(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(0, 0, 0, 5, 0, end_heading_deg=90.0)
        self.assertEqual(len(derivation.commands), 2)
        self.assertIs(derivation.commands[-1].command_type, CommandType.ROTATE)
        self.assertAlmostEqual(derivation.commands[-1].value, 90.0, delta=1e-4)

    def test_zero_distance_without_end_heading_produces_no_commands(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(3, 4, 0, 3, 4)
        self.assertEqual(len(derivation.commands), 0)

    def test_zero_distance_with_end_heading_produces_single_rotation(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(3, 4, 10, 3, 4, end_heading_deg=100)
        self.assertEqual(len(derivation.commands), 1)
        self.assertIs(derivation.commands[0].command_type, CommandType.ROTATE)
        self.assertAlmostEqual(derivation.commands[0].value, 90.0, delta=1e-4)

    def test_executing_derived_route_reaches_target_position(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(1, 1, 20, -4, 8)
        rover = Rover(initial_x=1, initial_y=1, initial_heading_deg=20)
        rover.run_program(derivation.commands)
        self.assertAlmostEqual(rover.state.x, -4.0, delta=1e-4)
        self.assertAlmostEqual(rover.state.y, 8.0, delta=1e-4)

    def test_executing_derived_route_reaches_target_heading(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(0, 0, 0, 6, 6, end_heading_deg=-45)
        rover = Rover()
        rover.run_program(derivation.commands)
        self.assertAlmostEqual(rover.state.x, 6.0, delta=1e-4)
        self.assertAlmostEqual(rover.state.y, 6.0, delta=1e-4)
        self.assertAlmostEqual(rover.state.heading_deg, -45.0, delta=1e-4)

    def test_normalize_degrees_wraps_correctly(self) -> None:
        self.assertAlmostEqual(TrajectoryPlanner.normalize_degrees(350), -10.0, delta=1e-9)
        self.assertAlmostEqual(TrajectoryPlanner.normalize_degrees(-190), 170.0, delta=1e-9)
        self.assertAlmostEqual(TrajectoryPlanner.normalize_degrees(0), 0.0, delta=1e-9)

    def test_invalid_end_point_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryPlanner.plan_route_to_point(0, 0, 0, float("nan"), 5)

    def test_format_route_derivation_contains_key_sections(self) -> None:
        derivation = TrajectoryPlanner.plan_route_to_point(0, 0, 0, 3, 4)
        text = format_route_derivation(derivation)
        self.assertIn("ROUTE PLANNING DERIVATION", text)
        self.assertIn("Displacement vector", text)
        self.assertIn("Distance to target", text)
        self.assertIn("Derived command sequence", text)


class TestInteractivePrompts(unittest.TestCase):
    """Interactive input helpers must validate, re-prompt, and support
    optional/default values correctly."""

    def test_prompt_float_uses_default_on_blank_input(self) -> None:
        with patch("builtins.input", return_value=""):
            value = prompt_float("Start X", default=2.5)
        self.assertEqual(value, 2.5)

    def test_prompt_float_parses_valid_number(self) -> None:
        with patch("builtins.input", return_value="7.5"):
            value = prompt_float("Start X")
        self.assertEqual(value, 7.5)

    def test_prompt_float_reprompts_on_invalid_input(self) -> None:
        with patch("builtins.input", side_effect=["abc", "3.0"]):
            value = prompt_float("Start X")
        self.assertEqual(value, 3.0)

    def test_prompt_optional_float_returns_none_on_blank(self) -> None:
        with patch("builtins.input", return_value=""):
            value = prompt_optional_float("End heading")
        self.assertIsNone(value)

    def test_prompt_optional_float_parses_value(self) -> None:
        with patch("builtins.input", return_value="45"):
            value = prompt_optional_float("End heading")
        self.assertEqual(value, 45.0)

    def test_prompt_start_and_end_points_full_flow(self) -> None:
        responses = iter(["1", "2", "30", "10", "20", "90"])
        def _mock_input(*_args: object) -> str:
            return next(responses)
        with patch("builtins.input", side_effect=_mock_input):
            result = prompt_start_and_end_points()
        self.assertEqual(result, (1.0, 2.0, 30.0, 10.0, 20.0, 90.0))

    def test_prompt_start_and_end_points_defaults_and_skipped_heading(self) -> None:
        responses = iter(["", "", "", "5", "5", ""])
        def _mock_input(*_args: object) -> str:
            return next(responses)
        with patch("builtins.input", side_effect=_mock_input):
            result = prompt_start_and_end_points()
        self.assertEqual(result, (0.0, 0.0, 0.0, 5.0, 5.0, None))

    def test_compass_direction_name(self) -> None:
        self.assertEqual(compass_direction_name(0), "East (+X)")
        self.assertEqual(compass_direction_name(90), "North (+Y)")
        self.assertEqual(compass_direction_name(180), "West (-X)")
        self.assertEqual(compass_direction_name(270), "South (-Y)")
        self.assertEqual(compass_direction_name(45), "North-East")


# ======================================================================
# Command-line interface
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rover Navigation Simulator - asks for a start point and an end "
            "point, derives the trajectory between them, and shows every "
            "matrix computation involved."
        )
    )
    parser.add_argument(
        "--start-x",
        type=float,
        default=0.0,
        metavar="X",
        help="Rover's starting X position (default: 0.0).",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=0.0,
        metavar="Y",
        help="Rover's starting Y position (default: 0.0).",
    )
    parser.add_argument(
        "--start-heading",
        type=float,
        default=0.0,
        metavar="DEGREES",
        help="Rover's starting heading in degrees (default: 0.0).",
    )
    parser.add_argument(
        "--end-x",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Target end X position. Given together with --end-y, this "
            "skips the interactive prompts and plans a route directly."
        ),
    )
    parser.add_argument(
        "--end-y",
        type=float,
        default=None,
        metavar="Y",
        help="Target end Y position.",
    )
    parser.add_argument(
        "--end-heading",
        type=float,
        default=None,
        metavar="DEGREES",
        help="Optional target end heading in degrees.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force the interactive start/end point prompts even if --end-x/--end-y are given.",
    )
    parser.add_argument(
        "--commands",
        type=str,
        default=None,
        metavar="PROGRAM",
        help=(
            "Legacy explicit-command mode: an inline command sequence, "
            "separated by ';' or newlines (e.g. \"FORWARD 5;ROTATE 90\"). "
            "Bypasses point-to-point planning entirely."
        ),
    )
    parser.add_argument(
        "--commands-file",
        type=str,
        default=None,
        help=(
            "Legacy explicit-command mode: path to a text file containing "
            "one command per line. Used if --commands is not given."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        metavar="PATH",
        help="File path to save the trajectory plot image (e.g. trajectory.png).",
    )
    parser.add_argument(
        "--show-calculations",
        action="store_true",
        help=(
            "Print the detailed, step-by-step matrix calculation report "
            "to console output (by default, calculations are saved to a file)."
        ),
    )
    parser.add_argument(
        "--calculations-file",
        type=str,
        default="matrix_operations.txt",
        metavar="PATH",
        help="Path to save the full derivation and matrix calculation report to a text file (defaults to matrix_operations.txt).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the built-in automated test suite instead of planning a route.",
    )
    return parser


def load_commands_from_file(path: str) -> List[str]:
    """Read raw command lines from a text file.

    Raises:
        FileNotFoundError: If the file cannot be opened or read.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readlines()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read commands file '{path}': {exc}") from exc


def resolve_inline_commands(raw_program: str) -> List[str]:
    """Split an inline, semicolon/newline-separated command string into
    individual raw command lines."""
    normalized = raw_program.replace("\n", ";")
    return [segment for segment in normalized.split(";")]


def resolve_raw_commands(args: argparse.Namespace) -> List[str]:
    """Determine the raw command lines to execute for explicit-command mode:
    --commands (inline) takes precedence over --commands-file."""
    if args.commands:
        return resolve_inline_commands(args.commands)
    if args.commands_file:
        return load_commands_from_file(args.commands_file)
    return DEMO_MISSION_COMMANDS


def run_mission(
    rover: Rover,
    commands: Sequence[Command],
    args: argparse.Namespace,
    preamble: Optional[str] = None,
) -> int:
    """Execute a command program on the given rover and handle all shared
    reporting and visualization output for both CLI modes.

    Returns:
        A process exit code (0 on success, 1 on a handled failure).
    """
    rover.run_program(commands)
    print_trajectory_table(rover)

    if args.show_calculations:
        print()
        print(MatrixCalculationReporter.generate_report(rover))

    if args.calculations_file:
        try:
            MatrixCalculationReporter.save_report(
                rover, args.calculations_file, preamble=preamble
            )
            print(f"\nMatrix calculation report saved to '{args.calculations_file}'")
        except OSError as exc:
            print(f"Error saving calculation report: {exc}", file=sys.stderr)
            return 1

    if MATPLOTLIB_AVAILABLE:
        try:
            TrajectoryVisualizer.plot(
                rover,
                save_path=args.save_plot,
                show=not args.no_show,
            )
        except RuntimeError as exc:
            print(f"Visualization error: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "Matplotlib not installed; skipping visualization. "
            "Install it with: pip install matplotlib",
            file=sys.stderr,
        )

    return 0


def run_explicit_command_mission(args: argparse.Namespace) -> int:
    """Legacy mode: execute a directly-supplied command sequence."""
    try:
        raw_commands = resolve_raw_commands(args)
    except FileNotFoundError as exc:
        print(f"Error loading commands: {exc}", file=sys.stderr)
        return 1

    try:
        commands = CommandParser.parse_program(raw_commands)
    except CommandParseError as exc:
        print(f"Error parsing commands: {exc}", file=sys.stderr)
        return 1

    if not commands:
        print("Error: no valid commands to execute.", file=sys.stderr)
        return 1

    try:
        rover = Rover(
            initial_x=args.start_x,
            initial_y=args.start_y,
            initial_heading_deg=args.start_heading,
        )
    except ValueError as exc:
        print(f"Error configuring start pose: {exc}", file=sys.stderr)
        return 1

    return run_mission(rover, commands, args)


def run_point_to_point_mission(args: argparse.Namespace) -> int:
    """Default mode: ask for (or read) a start and end point, derive the
    route between them, run it, and report every matrix step."""
    if args.end_x is not None and args.end_y is not None and not args.interactive:
        start_x, start_y, start_heading = args.start_x, args.start_y, args.start_heading
        end_x, end_y, end_heading = args.end_x, args.end_y, args.end_heading
    else:
        try:
            (
                start_x,
                start_y,
                start_heading,
                end_x,
                end_y,
                end_heading,
            ) = prompt_start_and_end_points()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted: no start/end point was provided.", file=sys.stderr)
            return 1

    try:
        derivation = TrajectoryPlanner.plan_route_to_point(
            start_x, start_y, start_heading, end_x, end_y, end_heading
        )
    except ValueError as exc:
        print(f"Error planning route: {exc}", file=sys.stderr)
        return 1

    derivation_text = format_route_derivation(derivation)
    print()
    print(derivation_text)

    try:
        rover = Rover(
            initial_x=start_x, initial_y=start_y, initial_heading_deg=start_heading
        )
    except ValueError as exc:
        print(f"Error configuring start pose: {exc}", file=sys.stderr)
        return 1

    return run_mission(rover, derivation.commands, args, preamble=derivation_text)


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args()

    if args.run_tests:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.commands or args.commands_file:
        return run_explicit_command_mission(args)

    return run_point_to_point_mission(args)


if __name__ == "__main__":
    sys.exit(main())