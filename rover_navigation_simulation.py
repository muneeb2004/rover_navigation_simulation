#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

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

Run this file directly for a demo mission and trajectory plot, or with
--run-tests to execute the full automated test suite.
"""

from __future__ import annotations

import argparse
import math
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:  # pragma: no cover - environment-dependent
    plt = None  # type: ignore
    _matplotlib_available = False

MATPLOTLIB_AVAILABLE: bool = _matplotlib_available


# ======================================================================
# Constants
# ======================================================================
NUMERICAL_TOLERANCE: float = 1e-6
TRAJECTORY_LINE_COLOR: str = "#4B2C6F"   # Habib University / OSL purple
HEADING_ARROW_COLOR: str = "#C9A24B"     # Habib University / OSL gold
START_MARKER_COLOR: str = "#2E7D32"
END_MARKER_COLOR: str = "#B00020"

# Default mission used by the CLI demo and by the chained-transformation test.
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

    def as_homogeneous_vector(self) -> np.ndarray[Any, Any]:
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
    def rotation_matrix(angle_rad: float) -> np.ndarray[Any, Any]:
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
    def translation_matrix(delta_x: float, delta_y: float) -> np.ndarray[Any, Any]:
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
    def combined_matrix(angle_rad: float, delta_x: float, delta_y: float) -> np.ndarray[Any, Any]:
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
    def local_transform_for_command(command: Command) -> np.ndarray[Any, Any]:
        """Build the local-frame transformation matrix for a single command."""
        if command.command_type is CommandType.FORWARD:
            return TransformationEngine.translation_matrix(command.value, 0.0)
        if command.command_type is CommandType.BACKWARD:
            return TransformationEngine.translation_matrix(-command.value, 0.0)
        if command.command_type is CommandType.ROTATE:
            return TransformationEngine.rotation_matrix(math.radians(command.value))
        raise ValueError(f"Unsupported command type: {command.command_type!r}")

    @staticmethod
    def compose_local_transforms(commands: Sequence[Command]) -> np.ndarray[Any, Any]:
        """Compose an ordered sequence of local-frame command transforms into
        a single composite matrix via chained matrix multiplication.

        The composite is built in chronological order (first command applied
        first): composite = T_1 @ T_2 @ ... @ T_n. Because each T_i is
        expressed in the rover's local frame at the moment it is issued,
        right-multiplication correctly accumulates world-frame pose.
        """
        composite = np.identity(3, dtype=np.float64)
        for command in commands:
            composite = composite @ TransformationEngine.local_transform_for_command(command)
        return composite

    @staticmethod
    def extract_state_from_world_transform(matrix: np.ndarray[Any, Any]) -> RoverState:
        """Recover (x, y, heading) from a 3x3 homogeneous world transform."""
        x, y = float(matrix[0, 2]), float(matrix[1, 2])
        heading_rad = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        return RoverState(x=x, y=y, heading_rad=heading_rad)


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
        self._world_transform: np.ndarray[Any, Any] = TransformationEngine.combined_matrix(
            heading_rad, x, y
        )
        self.state: RoverState = RoverState(x=x, y=y, heading_rad=heading_rad)
        self.trajectory: List[RoverState] = [self.state.copy()]

    @staticmethod
    def _validate_finite(value: Any, name: str) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"'{name}' must be a finite number, got {value!r}.")

    def apply_command(self, command: Command) -> RoverState:
        """Apply a single command, update world pose, and record trajectory."""
        if not isinstance(command, Command):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected a Command instance, got {type(command).__name__}.")

        local_transform = TransformationEngine.local_transform_for_command(command)
        self._world_transform = self._world_transform @ local_transform
        self.state = TransformationEngine.extract_state_from_world_transform(
            self._world_transform
        )
        self.trajectory.append(self.state.copy())
        return self.state

    def run_program(self, commands: Sequence[Command]) -> RoverState:
        """Apply an ordered sequence of commands, returning the final state."""
        for command in commands:
            self.apply_command(command)
        return self.state

    def get_trajectory_arrays(self) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
        if len(rover.trajectory) == 0:
            raise ValueError("Rover has no recorded trajectory to plot.")

        xs, ys, thetas = rover.get_trajectory_arrays()

        figure, axes = plt.subplots(figsize=(8, 8))
        axes.plot(
            xs, ys,
            color=TRAJECTORY_LINE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Trajectory",
            zorder=2,
        )
        axes.scatter(
            [xs[0]], [ys[0]], color=START_MARKER_COLOR, s=90, zorder=5, label="Start"
        )
        axes.scatter(
            [xs[-1]], [ys[-1]], color=END_MARKER_COLOR, s=90, zorder=5, label="End"
        )

        arrow_length = TrajectoryVisualizer._compute_arrow_length(xs, ys)
        stride = max(1, arrow_stride)
        for index in range(0, len(xs), stride):
            dx = arrow_length * math.cos(thetas[index])
            dy = arrow_length * math.sin(thetas[index])
            axes.annotate(
                "",
                xy=(xs[index] + dx, ys[index] + dy),
                xytext=(xs[index], ys[index]),
                arrowprops=dict(arrowstyle="->", color=HEADING_ARROW_COLOR, lw=1.5),
                zorder=4,
            )

        axes.set_title(title, fontsize=14, fontweight="bold")
        axes.set_xlabel("X Position")
        axes.set_ylabel("Y Position")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, linestyle="--", alpha=0.4)
        axes.legend(loc="best")

        if save_path:
            figure.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)

    @staticmethod
    def _compute_arrow_length(xs: np.ndarray[Any, Any], ys: np.ndarray[Any, Any]) -> float:
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


# ======================================================================
# Command-line interface
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rover Navigation Simulator - computes and visualizes a rover's "
            "trajectory from a sequence of movement commands."
        )
    )
    parser.add_argument(
        "--commands-file",
        type=str,
        default=None,
        help=(
            "Path to a text file containing one command per line "
            "(e.g. 'FORWARD 5'). Defaults to a built-in demo mission."
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
        "--run-tests",
        action="store_true",
        help="Run the built-in automated test suite instead of the demo mission.",
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


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args()

    if args.run_tests:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    raw_commands = (
        load_commands_from_file(args.commands_file)
        if args.commands_file
        else DEMO_MISSION_COMMANDS
    )

    try:
        commands = CommandParser.parse_program(raw_commands)
    except CommandParseError as exc:
        print(f"Error parsing commands: {exc}", file=sys.stderr)
        return 1

    if not commands:
        print("Error: no valid commands to execute.", file=sys.stderr)
        return 1

    rover = Rover()
    rover.run_program(commands)
    print_trajectory_table(rover)

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


if __name__ == "__main__":
    sys.exit(main())