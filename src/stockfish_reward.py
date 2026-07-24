#!/usr/bin/env python
"""
Stockfish Reward Wrapper for ChessRL

Provides evaluation functions to compute rewards using Stockfish engine.
Uses the stockfish Python library.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from stockfish import Stockfish
except ImportError:
    print("Please install stockfish: pip install stockfish")
    raise


class StockfishReward:
    """Wrapper around Stockfish engine for chess reward computation."""

    def __init__(self, path: str = "stockfish", depth: int = 15, threads: int = 1):
        """
        Initialize Stockfish wrapper.

        Args:
            path: Path to stockfish executable
            depth: Search depth (higher = more accurate but slower)
            threads: Number of CPU threads to use
        """
        self.path = path
        self.depth = depth
        self.threads = threads
        self.stockfish = Stockfish(
            path=path,
            depth=depth,
            parameters={"Threads": threads}
        )

    def set_position(self, moves: List[str]) -> str:
        """
        Set position with a list of moves.

        Args:
            moves: List of UCI moves (e.g., ["e2e4", "e7e5"])

        Returns:
            Current FEN position
        """
        if not moves:
            # Empty moves - return starting position
            self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            return self.stockfish.get_fen_position()

        try:
            # Use make_moves_from_current_position in a loop instead of make_moves_from_start
            # make_moves_from_start doesn't work reliably with the stockfish library
            for move in moves:
                self.stockfish.make_moves_from_current_position([move])
        except (ValueError, AttributeError) as e:
            # If moves are invalid, reset to starting position
            self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        return self.stockfish.get_fen_position()

    def get_best_move(self) -> str:
        """Get best move from current position."""
        return self.stockfish.get_best_move()

    def get_evaluation(self) -> Dict[str, Any]:
        """
        Get evaluation of current position.

        Returns:
            Dict with 'type' (cp/mate) and 'value' (score)
        """
        return self.stockfish.get_evaluation()

    def get_top_moves(self, num_moves: int = 3) -> List[Dict]:
        """
        Get top N best moves with their evaluations.

        Args:
            num_moves: Number of moves to return

        Returns:
            List of dicts with 'Move', 'Score', 'Mate', 'Depth'
        """
        self.stockfish.update_engine_parameters({"MultiPV": num_moves})
        return self.stockfish.get_top_moves()

    def _parse_eval_score(self, eval_result: Dict) -> int:
        """Parse evaluation score from Stockfish result."""
        if eval_result is None:
            return 0
        if eval_result.get("type") == "mate":
            return 10000 + eval_result["value"]
        return eval_result.get("value", 0)

    def close(self):
        """Close Stockfish process."""
        try:
            self.stockfish.send_quit_command()
        except Exception:
            pass


# Global instance
_stockfish_reward: Optional[StockfishReward] = None


def get_stockfish(path: str = "stockfish", depth: int = 15, threads: int = 1) -> StockfishReward:
    """Get or create global Stockfish instance."""
    global _stockfish_reward
    if _stockfish_reward is None:
        _stockfish_reward = StockfishReward(path=path, depth=depth, threads=threads)
    return _stockfish_reward


def reset_stockfish():
    """Reset global Stockfish instance."""
    global _stockfish_reward
    if _stockfish_reward:
        _stockfish_reward.close()
    _stockfish_reward = None


# Reward computation functions


def _extract_json_from_prediction(prediction: str) -> str:
    """
    Extract JSON from prediction that may contain thinking tokens.

    Args:
        prediction: Model's prediction string (may contain <think>/ tokens)

    Returns:
        Extracted JSON string or original prediction if no JSON found
    """
    import re

    # Try to find a JSON object in the prediction
    # Look for patterns like {"key": "value"} - non-greedy to avoid matching too much
    # This handles nested braces up to a reasonable depth
    for depth in range(1, 10):
        pattern = r'\{[^{}]*\}' * depth
        json_match = re.search(pattern, prediction)
        if json_match:
            return json_match.group()

    # Fallback: try simpler pattern matching any JSON-like structure
    # Look for opening brace, then capture until we find a balanced closing brace
    brace_count = 0
    start_idx = prediction.find('{')
    if start_idx >= 0:
        for i, char in enumerate(prediction[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return prediction[start_idx:i+1]

    # Final fallback: return original
    return prediction


def _extract_move_from_prediction(prediction: str) -> str:
    """
    Extract move from prediction that may be JSON or plain text.

    Args:
        prediction: Model's prediction string

    Returns:
        Extracted move string
    """
    import re

    # First try to extract JSON and get the move from there
    json_str = _extract_json_from_prediction(prediction)
    try:
        pred_json = json.loads(json_str.replace("'", '"'))
        # Try various possible keys (case-insensitive)
        for key in ['next best move', 'missing move', 'move', 'best move']:
            if key.lower() in [k.lower() for k in pred_json.keys()]:
                # Find the actual key
                for k in pred_json.keys():
                    if k.lower() == key.lower():
                        return str(pred_json[k]).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try to extract UCI move pattern from text
    clean_pred = prediction.strip().strip('"').strip("'").strip()
    move_pattern = re.search(r'[a-h][1-8][a-h][1-8]', clean_pred)
    if move_pattern:
        return move_pattern.group()

    return ""


def _score_to_reward(score: int, side_to_move_is_white: bool = True) -> float:
    """
    Convert Stockfish centipawn score to a normalized reward.
    Positive score = advantage for side to move.

    For reward computation, we typically want to measure White's advantage.
    If side_to_move_is_white=True, positive score = White advantage.
    If side_to_move_is_white=False, we need to flip the score (Black moving = White disadvantage).

    Args:
        score: Stockfish evaluation score (positive = side to move advantage)
        side_to_move_is_white: Whether White is to move

    Returns:
        Reward value (-1.0 to 1.0) where positive = White advantage
    """
    # Clamp score to reasonable range (-5000 to 5000 cp)
    score = max(-5000, min(5000, score))

    # If Black is to move, flip the score (Black advantage = White disadvantage)
    if not side_to_move_is_white:
        score = -score

    # Normalize to -1.0 to 1.0 range
    # 100 cp = 0.1 pawn advantage, 1000 cp = 1 pawn
    # Scale to make rewards more meaningful
    normalized = score / 500.0  # 500 cp = 0.5 pawn = 1.0 reward

    # Clamp to -1.0 to 1.0
    return max(-1.0, min(1.0, normalized))


def _score_to_reward_scaled(score: int, side_to_move_is_white: bool = True) -> float:
    """
    Convert Stockfish centipawn score to a scaled reward for RL.
    This provides more differentiation between moves.
    """
    # Clamp score to reasonable range (-5000 to 5000 cp)
    score = max(-5000, min(5000, score))

    # If Black is to move, flip the score (Black advantage = White disadvantage)
    if not side_to_move_is_white:
        score = -score

    # Scale to -1.0 to 1.0 with more granularity
    # Use tanh-like scaling for smooth gradients across the range
    import math
    normalized = math.tanh(score / 200.0)  # 200 cp = 1 pawn = tanh(1) ~ 0.76

    # Scale to -1.0 to 1.0
    return max(-1.0, min(1.0, normalized))


def compute_reward_find_next_best_move(
    prediction: str, stockfish: StockfishReward, moves: List[str]
) -> float:
    """
    Compute reward for FIND_NEXT_BEST_MOVE category.

    Uses Stockfish's evaluation score as a continuous reward signal.
    The reward is based on the evaluation of the position after playing
    the predicted move.

    Args:
        prediction: Predicted move string
        stockfish: StockfishReward instance
        moves: List of moves to set position

    Returns:
        Reward value (-1.0 to 1.0)
    """
    try:
        predicted_move = _extract_move_from_prediction(prediction)

        # Set position
        stockfish.set_position(moves)

        # Make the predicted move
        stockfish.stockfish.make_moves_from_current_position([predicted_move])

        # Get the evaluation of the resulting position
        eval_result = stockfish.get_evaluation()
        eval_score = eval_result.get("value", 0)

        # Determine whose turn it is in the new position
        # FEN format: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # The 'w' or 'b' indicates who is to move
        new_fen = stockfish.stockfish.get_fen_position()
        side_to_move_is_white = ' w ' in new_fen

        # Convert score to reward (White advantage)
        reward = _score_to_reward_scaled(eval_score, side_to_move_is_white)

        return reward
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        # If move is invalid or error occurs, give a small penalty
        return -0.1


def compute_reward_find_advantaged_player(
    prediction: str, stockfish: StockfishReward, moves: List[str]
) -> float:
    """
    Compute reward for FIND_ADVANTAGED_PLAYER category.

    Uses Stockfish's evaluation score as a continuous reward signal.
    Reward is based on how accurately the prediction matches the eval
    and the strength of the advantage.

    Args:
        prediction: Predicted advantage ("White", "Black", or "Draw")
        stockfish: StockfishReward instance
        moves: List of moves to set position

    Returns:
        Reward value (-1.0 to 1.0)
    """
    try:
        # Extract JSON from prediction (may contain thinking tokens)
        json_str = _extract_json_from_prediction(prediction)

        # Try to parse as JSON first
        try:
            pred_json = json.loads(json_str.replace("'", '"'))
            # Normalize key lookup to lowercase
            predicted_advantage = ""
            for key, value in pred_json.items():
                if "advantage" in key.lower() and "most" in key.lower():
                    predicted_advantage = str(value).strip()
                    break
        except json.JSONDecodeError:
            # If not JSON, extract advantage from text
            predicted_advantage = ""
            if "white" in prediction.lower() and "black" not in prediction.lower():
                predicted_advantage = "white"
            elif "black" in prediction.lower() and "white" not in prediction.lower():
                predicted_advantage = "black"
            elif "draw" in prediction.lower():
                predicted_advantage = "draw"
            else:
                # Try to find any advantage mention
                import re
                match = re.search(r'white|black|draw', prediction, re.IGNORECASE)
                if match:
                    predicted_advantage = match.group().lower()

        # Set position
        stockfish.set_position(moves)

        # Get evaluation score
        eval_result = stockfish.get_evaluation()
        eval_score = eval_result.get("value", 0)

        # Convert eval to advantage prediction (normalize to lowercase)
        # Use smaller thresholds (> 0.5 pawn = 50 cp) for more leniency
        if eval_score > 50:
            expected = "white"
        elif eval_score < -50:
            expected = "black"
        else:
            expected = "draw"

        # Normalize predicted advantage to lowercase for comparison
        predicted_advantage = predicted_advantage.lower()

        # Compute reward based on accuracy AND strength of advantage
        # Scale by eval strength for more granular feedback
        score_scaled = _score_to_reward_scaled(eval_score, True)  # Always scale for White advantage

        if predicted_advantage == expected:
            # Scale reward by eval strength
            if abs(eval_score) > 200:
                return max(0.5, score_scaled)  # Strong advantage, correctly predicted
            elif abs(eval_score) > 50:
                return max(0.2, score_scaled * 0.5)  # Moderate advantage
            elif abs(eval_score) > 25:
                return max(0.0, score_scaled * 0.3)  # Weak advantage
            else:
                return max(-0.1, score_scaled * 0.2)  # Very weak
        else:
            # Wrong prediction - penalty based on how wrong
            if predicted_advantage == "Draw" or expected == "Draw":
                return -0.2  # Close call - partial credit
            else:
                return max(-0.5, -abs(score_scaled))  # Wrong prediction
    except (json.JSONDecodeError, AttributeError):
        return -1.0


def compute_reward_find_final_score(
    prediction: str, stockfish: StockfishReward, moves: List[str]
) -> float:
    """
    Compute reward for FIND_FINAL_SCORE category.

    Uses Stockfish's evaluation score as a continuous reward signal.
    Reward is based on how accurately the prediction matches the eval
    and the strength of the advantage.

    Args:
        prediction: Predicted score ("1-0", "0-1", or "1/2-1/2")
        stockfish: StockfishReward instance
        moves: List of moves

    Returns:
        Reward value (-1.0 to 1.0)
    """
    try:
        # Extract JSON from prediction (may contain thinking tokens)
        json_str = _extract_json_from_prediction(prediction)

        # Try to parse as JSON first
        try:
            pred_json = json.loads(json_str.replace("'", '"'))
            # Normalize key lookup for score
            predicted_score = ""
            for key, value in pred_json.items():
                if "score" in key.lower():
                    predicted_score = str(value).strip()
                    break
        except json.JSONDecodeError:
            # If not JSON, maybe it's just the score directly (like "1-0")
            predicted_score = prediction.strip().strip('"').strip("'").strip()

        # Set position
        stockfish.set_position(moves)

        # Get evaluation
        eval_result = stockfish.get_evaluation()
        eval_score = eval_result.get("value", 0)

        # Convert eval to expected score (normalized to lowercase)
        if eval_score > 200:
            expected = "1-0"
        elif eval_score < -200:
            expected = "0-1"
        else:
            expected = "1/2-1/2"

        # Normalize predicted score to lowercase for comparison
        predicted_score = predicted_score.lower().strip()

        # Compute reward based on accuracy AND eval strength
        score_scaled = _score_to_reward_scaled(eval_score, True)  # Scale for White advantage

        if predicted_score == expected:
            # Scale reward by eval strength for more granular feedback
            if abs(eval_score) > 400:
                return max(0.5, score_scaled)
            elif abs(eval_score) > 200:
                return max(0.3, score_scaled * 0.8)
            elif abs(eval_score) > 100:
                return max(0.1, score_scaled * 0.5)
            else:
                return max(-0.1, score_scaled * 0.3)
        else:
            # Wrong prediction
            if predicted_score == "1/2-1/2" or expected == "1/2-1/2":
                return -0.15  # Close call - partial credit
            else:
                return max(-0.5, -abs(score_scaled))
    except (json.JSONDecodeError, AttributeError):
        return -1.0


def compute_reward_mlm_on_moves(
    prediction: str, stockfish: StockfishReward, moves: List[str]
) -> float:
    """
    Compute reward for MLM_ON_MOVES category.

    Args:
        prediction: Predicted missing moves
        stockfish: StockfishReward instance
        moves: List of moves with missing ones marked

    Returns:
        Reward value (0.0 to 1.0 per correct move)
    """
    try:
        # Extract JSON from prediction (may contain thinking tokens)
        json_str = _extract_json_from_prediction(prediction)
        # Parse prediction - normalize key lookups
        pred_json = json.loads(json_str.replace("'", '"'))
        missing_moves_str = ""
        for key, value in pred_json.items():
            if "missing" in key.lower() and "move" in key.lower():
                missing_moves_str = value
                break

        # Parse missing moves
        if isinstance(missing_moves_str, list):
            predicted_moves = [m.strip().strip("[]\"'") for m in missing_moves_str if m.strip()]
        else:
            # Try to parse as comma-separated
            predicted_moves = []
            for part in missing_moves_str.replace("[", "").replace("]", "").split(","):
                part = part.strip().strip("\"'")
                if part and part != "?":
                    predicted_moves.append(part)

        if not predicted_moves:
            return -1.0

        # For each predicted move, verify it's a good move
        correct_count = 0
        total_moves = len(predicted_moves)

        for move in predicted_moves:
            stockfish.set_position(moves)
            best_move = stockfish.get_best_move()

            # Check if predicted move is in top Stockfish options (5 instead of 3)
            top_moves = stockfish.get_top_moves(num_moves=5)
            top_move_list = [m.get("Move", "") for m in top_moves]

            if move in top_move_list or move == best_move:
                correct_count += 1

        # Scale reward to be between -0.5 and 1.0
        if total_moves == 0:
            return -1.0
        base_reward = correct_count / total_moves
        return 0.5 + 0.5 * base_reward  # Maps 0->0.5, 1->1.0
    except (json.JSONDecodeError, AttributeError):
        return -1.0


def compute_reward_find_last_move(
    prediction: str, stockfish: StockfishReward, moves: List[str]
) -> float:
    """
    Compute reward for FIND_LAST_MOVE category.

    Args:
        prediction: Predicted last move
        stockfish: StockfishReward instance
        moves: List of moves

    Returns:
        Reward value (-1.0 to 1.0)
    """
    try:
        predicted_move = _extract_move_from_prediction(prediction)

        # Set position
        stockfish.set_position(moves)

        # Get best move
        best_move = stockfish.get_best_move()

        # Check if predicted move is in top 5 moves
        top_moves = stockfish.get_top_moves(num_moves=5)
        top_move_list = [m.get("Move", "") for m in top_moves]

        if not predicted_move:
            return -0.5  # No move found

        if predicted_move == best_move:
            return 1.0
        elif predicted_move in top_move_list:
            return 0.5
        else:
            return -0.25
    except (json.JSONDecodeError, AttributeError):
        return -1.0


def compute_reward_sort_fens(
    prediction: str, stockfish: StockfishReward, fens: List[str]
) -> float:
    """
    Compute reward for SORT_FENS category.

    Args:
        prediction: Predicted sorted FENs
        stockfish: StockfishReward instance
        fens: Original FENs

    Returns:
        Reward value (-1.0 to 1.0)
    """
    try:
        # Extract JSON from prediction (may contain thinking tokens)
        json_str = _extract_json_from_prediction(prediction)
        # Parse prediction - normalize key lookup
        pred_json = json.loads(json_str.replace("'", '"'))
        sorted_fens_str = ""
        for key, value in pred_json.items():
            if "sorted" in key.lower() and "fen" in key.lower():
                sorted_fens_str = value
                break

        if isinstance(sorted_fens_str, str):
            sorted_fens = [f.strip().strip("\"'") for f in sorted_fens_str.strip("[]").split(",") if f.strip()]
        elif isinstance(sorted_fens_str, list):
            sorted_fens = [str(f).strip("\"'") for f in sorted_fens_str if str(f).strip()]
        else:
            return -1.0

        if not sorted_fens or len(sorted_fens) < 2:
            return -1.0

        # Verify ordering by checking eval increases for white
        prev_eval = float("-inf")
        valid_count = 0

        for fen in sorted_fens:
            try:
                stockfish.stockfish.set_fen_position(fen)
                eval_result = stockfish.get_evaluation()
                eval_score = eval_result.get("value", 0)

                if eval_score >= prev_eval:
                    valid_count += 1
                prev_eval = eval_score
            except Exception:
                continue

        # Reward based on correct ordering - scale to range [-0.5, 1.0]
        if sorted_fens:
            return 0.5 + 0.5 * (valid_count / len(sorted_fens))
        return -1.0
    except (json.JSONDecodeError, AttributeError):
        return -1.0


def compute_reward(
    prediction: str,
    category: str,
    stockfish: StockfishReward,
    moves: Optional[List[str]] = None,
    fens: Optional[List[str]] = None,
) -> float:
    """
    Compute reward based on task category.

    Args:
        prediction: Model's predicted output string
        category: Task category (KIND)
        stockfish: StockfishReward instance
        moves: List of moves for position
        fens: List of FENs for sorting task

    Returns:
        Reward value
    """
    if category == "FIND_NEXT_BEST_MOVE":
        return compute_reward_find_next_best_move(prediction, stockfish, moves or [])
    elif category == "FIND_ADVANTAGED_PLAYER":
        return compute_reward_find_advantaged_player(prediction, stockfish, moves or [])
    elif category == "FIND_FINAL_SCORE":
        return compute_reward_find_final_score(prediction, stockfish, moves or [])
    elif category == "MLM_ON_MOVES":
        return compute_reward_mlm_on_moves(prediction, stockfish, moves or [])
    elif category == "FIND_LAST_MOVE":
        return compute_reward_find_last_move(prediction, stockfish, moves or [])
    elif category == "SORT_FENS":
        return compute_reward_sort_fens(prediction, stockfish, fens or [])
    else:
        return 0.0


if __name__ == "__main__":
    # Test Stockfish wrapper
    sf = StockfishReward(path="/hfcache/harissh/ChessLM/stockfish")
    sf.set_position(["e2e4", "e7e5"])
    print(f"Position FEN: {sf.stockfish.get_fen_position()}")
    print(f"Best move: {sf.get_best_move()}")
    print(f"Evaluation: {sf.get_evaluation()}")
    print(f"Top 3 moves: {sf.get_top_moves(3)}")