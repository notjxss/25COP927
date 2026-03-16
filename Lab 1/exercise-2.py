"""rps_lab1_part1_random.py

Fundamentals of AI — Lab 1 (Part 1)
==================================

Goal
----
Play Rock–Paper–Scissors (RPS) against a computer that chooses *randomly*.

Why this file exists
--------------------
This is the **baseline** program students start from. It is intentionally simple:

* Input validation (only accept rock/paper/scissors)
* Random choice for the computer
* Clear calculation of win / loss / tie
* A small "histogram" (counts) printed at the end

Students will later extend this baseline with Bayesian learning.

How to run
----------
python rps_lab1_part1_random.py

Tip for the lab
---------------
Set NUM_ROUNDS to 15 to match the lab instructions.
"""

# ---------------------------
# Imports
# ---------------------------

import random  # used to generate the computer's random move


# ---------------------------
# Configuration (easy for beginners to edit)
# ---------------------------

NUM_ROUNDS = 15  # how many rounds to play in one session


# ---------------------------
# Data definitions
# ---------------------------

# The three valid moves in Rock–Paper–Scissors.
MOVES = ("rock", "paper", "scissors")

# Map short inputs to full move names.
# Students often prefer typing r / p / s.
SHORT_TO_MOVE = {
    "r": "rock",
    "p": "paper",
    "s": "scissors",
}

# For display: which move beats which?
# Example: paper beats rock.
BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


# ---------------------------
# Helper functions
# ---------------------------

def get_user_move() -> str:
    """Ask the user for a move and return it as a full word.

    Returns
    -------
    str
        One of: "rock", "paper", "scissors".
    """

    # Keep asking until the user provides a valid input.
    while True:
        # We accept both short (r/p/s) and full words.
        raw = input("Choose [r]ock, [p]aper, [s]cissors: ").strip().lower()

        # Convert short input to a full move (if possible).
        if raw in SHORT_TO_MOVE:
            return SHORT_TO_MOVE[raw]

        # If the user typed the full move name, accept it.
        if raw in MOVES:
            return raw

        # If we get here, the input was invalid.
        print("Invalid input. Please type r/p/s or rock/paper/scissors.")


def get_computer_move_random() -> str:
    """Return a random move for the computer."""

    # random.choice picks one element uniformly at random.
    return random.choice(MOVES)


def outcome_from_user_perspective(user_move: str, computer_move: str) -> str:
    """Compute the round outcome from the user's perspective.

    Parameters
    ----------
    user_move : str
        The user's move ("rock"/"paper"/"scissors").
    computer_move : str
        The computer's move ("rock"/"paper"/"scissors").

    Returns
    -------
    str
        "win"  if the user wins,
        "loss" if the user loses,
        "tie"  if both chose the same move.
    """

    # If both moves are the same, it's a tie.
    if user_move == computer_move:
        return "tie"

    # The user wins if their move beats the computer's move.
    # Example: user_move="paper" beats "rock" because BEATS["paper"] == "rock".
    if BEATS[user_move] == computer_move:
        return "win"

    # Otherwise the user loses.
    return "loss"


def print_round_summary(round_index: int, user_move: str, computer_move: str, outcome: str) -> None:
    """Print what happened in a single round."""

    # Human-friendly mapping from outcome label to a message.
    message = {
        "win": "You win! ✅",
        "loss": "Computer wins. ❌",
        "tie": "It's a tie. 🤝",
    }[outcome]

    print(f"\nRound {round_index}:")
    print(f"  You:      {user_move}")
    print(f"  Computer: {computer_move}")
    print(f"  Result:   {message}")


def print_session_summary(counts: dict[str, int]) -> None:
    """Print the end-of-session histogram (wins/losses/ties)."""

    total = counts["win"] + counts["loss"] + counts["tie"]

    # Avoid division by zero (shouldn't happen if total>0, but safe anyway).
    if total == 0:
        print("No rounds were played.")
        return

    # Convert counts to percentages for an easy-to-read summary.
    win_rate = counts["win"] / total
    loss_rate = counts["loss"] / total
    tie_rate = counts["tie"] / total

    print("\n============================")
    print("Session summary")
    print("============================")
    print(f"Rounds played: {total}")
    print(f"Wins:  {counts['win']:>3}  ({win_rate:.2%})")
    print(f"Losses:{counts['loss']:>3}  ({loss_rate:.2%})")
    print(f"Ties:  {counts['tie']:>3}  ({tie_rate:.2%})")


def update_outcome_counts(counts: dict[str, int], outcome: str) -> None:
    counts[outcome] += 1

def predict_next_outcome(counts: dict[str,int]):
    alpha_prior = {
        "win": 1,
        "loss": 1,
        "tie": 1
    }

    alpha_post = {
        "win": alpha_prior["win"] + counts["win"],
        "loss": alpha_prior["loss"] + counts["loss"],
        "tie": alpha_prior["tie"] + counts["tie"]
    }

    total = sum(alpha_post.values())

    return {
        "win": alpha_post["win"] / total,
        "loss": alpha_post["loss"] / total,
        "tie": alpha_post["tie"] / total
    }

# ---------------------------
# Main program
# ---------------------------

def main() -> None:
    """Run one RPS session of NUM_ROUNDS rounds."""

    print("\nRock–Paper–Scissors (baseline: random computer)")
    print("------------------------------------------------")
    print(f"We will play {NUM_ROUNDS} rounds.\n")

    # Histogram (counts) of outcomes.
    # We store counts as integers.
    counts = {"win": 0, "loss": 0, "tie": 0}

    # Play NUM_ROUNDS rounds.
    for i in range(1, NUM_ROUNDS + 1):
        user_move = get_user_move()  # ask the user for their move
        computer_move = get_computer_move_random()  # choose randomly
        outcome = outcome_from_user_perspective(user_move, computer_move)  # win/loss/tie

        # Print what happened this round.
        print_round_summary(i, user_move, computer_move, outcome)
        update_outcome_counts(counts, outcome)
        probability = predict_next_outcome(counts)

        print(f"Beliefs about next round: ")
        print(f"P(win): {probability['win']:.2f}")
        print(f"P(loss): {probability['loss']:.2f}")
        print(f"P(tie): {probability['tie']:.2f}")

    # Print the final histogram.
    print_session_summary(counts)


# This line makes sure main() runs only when we execute this file directly.
if __name__ == "__main__":
    main()
