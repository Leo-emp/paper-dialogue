"""
Paper Dialogue: Two AI agents debate a research paper.
- Skeptic: Finds flaws in methodology, statistical reasoning, and claims.
- Visionary: Identifies future applications, extensions, and potential impact.
- Moderator: Opens the dialogue and synthesizes the final debate.

Uses Google Gemini API (free tier — no payment required).
Get your free API key at: https://aistudio.google.com/apikey
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse      # For parsing command-line arguments (--rounds, --model, etc.)
import json          # For saving dialogue output as JSON
import os            # For reading environment variables (API key)
import sys           # For exiting the program on errors
from pathlib import Path  # For file path operations and checking if PDF exists

from google import genai         # Google Gemini AI SDK
from google.genai import types   # Type definitions for Gemini API calls
import fitz                      # PyMuPDF — for extracting text from PDF files


# ============================================================
# AGENT SYSTEM PROMPTS
# These define the personality and behavior of each agent.
# The system prompt is sent to the AI model before the
# conversation starts, telling it how to behave.
# ============================================================

# The Skeptic agent — finds flaws and weaknesses in the paper
SKEPTIC_SYSTEM = """You are the Skeptic — a rigorous, methodical AI researcher who critically examines research papers. Your role in this dialogue:

- Challenge the paper's methodology: sample sizes, controls, statistical tests, baselines, ablations.
- Question assumptions and hidden biases in the experimental setup.
- Identify logical leaps, overclaimed results, and gaps between evidence and conclusions.
- Point out missing comparisons, reproducibility concerns, and potential confounders.
- Be specific — cite sections, tables, or figures from the paper when making critiques.

Tone: Sharp but fair. You're not hostile — you genuinely want the science to be better. You acknowledge strengths before dissecting weaknesses.

You are in a dialogue with the Visionary. Respond to their points directly, push back where you disagree, and concede when they make a valid argument. Keep responses focused (2-4 paragraphs)."""

# The Visionary agent — finds future potential and applications
VISIONARY_SYSTEM = """You are the Visionary — a forward-thinking AI researcher who sees the potential in research papers. Your role in this dialogue:

- Identify promising future applications and extensions of the work.
- Connect the paper's contributions to broader trends in AI/ML.
- Propose how the techniques could be combined with other methods or applied to new domains.
- Discuss societal impact, both positive risks and opportunities.
- Acknowledge limitations the Skeptic raises, but reframe them as open problems worth solving.

Tone: Enthusiastic but grounded. You back your optimism with technical reasoning, not hype. You respect the Skeptic's critiques and use them to refine your vision.

You are in a dialogue with the Skeptic. Respond to their points directly, build on valid critiques, and defend your position when you believe the potential is real. Keep responses focused (2-4 paragraphs)."""

# The Moderator agent — introduces and wraps up the dialogue
MODERATOR_SYSTEM = """You are the Moderator of an academic dialogue about a research paper. Your jobs:

1. OPENING: Write a brief (2-3 sentence) introduction of the paper and the dialogue format.
2. CLOSING: After the dialogue rounds, write a synthesis (3-5 paragraphs) that:
   - Summarizes the key tensions and agreements
   - Identifies which critiques were most compelling
   - Identifies which future directions were most promising
   - Gives a balanced final assessment

Be concise and specific. Reference actual points made during the dialogue."""


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path: str, max_pages: int = 50) -> str:
    """
    Reads a PDF file and extracts all text from it.

    Args:
        pdf_path: Path to the PDF file on disk.
        max_pages: Maximum number of pages to read (default 50).
                   This prevents extremely long papers from using
                   too many API tokens.

    Returns:
        A single string containing all text from the PDF.
    """
    # Open the PDF file using PyMuPDF
    doc = fitz.open(pdf_path)

    # Extract text from each page, up to the max_pages limit
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pages.append(page.get_text())  # get_text() extracts all text from a page

    doc.close()

    # Join all pages into one big string, separated by double newlines
    text = "\n\n".join(pages)

    # If the text is too long (over 80k chars), truncate it
    # to stay within API token limits
    if len(text) > 80_000:
        text = text[:80_000] + "\n\n[...truncated for length]"

    return text


# ============================================================
# API CALL HELPER
# ============================================================

def call_agent(client: genai.Client, system: str, messages: list[dict], model: str) -> str:
    """
    Sends a conversation to the Gemini API and returns the AI's response.

    Args:
        client:   The Gemini API client (authenticated).
        system:   The system prompt — defines the agent's personality.
        messages: The conversation history as a list of dicts.
                  Each dict has "role" ("user" or "model") and "content" (text).
        model:    The Gemini model name to use (e.g., "gemini-2.0-flash").

    Returns:
        The AI-generated text response as a string.
    """
    # Convert our simple message format into Gemini's Content objects
    # Gemini uses "user" and "model" as role names
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Make the API call to Gemini
    response = client.models.generate_content(
        model=model,           # Which Gemini model to use
        contents=contents,     # The conversation history
        config=types.GenerateContentConfig(
            system_instruction=system,    # The agent's personality/instructions
            max_output_tokens=1500,       # Limit response length
        ),
    )

    # Return just the text from the response
    return response.text


# ============================================================
# MAIN DIALOGUE ORCHESTRATOR
# This is the core function that runs the entire dialogue.
# ============================================================

def run_dialogue(pdf_path: str, rounds: int = 3, model: str = "gemini-2.0-flash", output_file: str | None = None):
    """
    Orchestrates the full Skeptic vs. Visionary dialogue.

    Flow:
        1. Extract text from the PDF
        2. Moderator gives an opening introduction
        3. For each round:
           a. Skeptic critiques the paper
           b. Visionary responds to the critique
           c. Each agent sees the other's messages for the next round
        4. Moderator gives a closing synthesis
        5. Optionally save to file

    Args:
        pdf_path:    Path to the research paper PDF.
        rounds:      Number of back-and-forth rounds (default 3).
        model:       Gemini model to use.
        output_file: Optional path to save the dialogue (.txt or .json).
    """

    # --- Step 1: Check for API key ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY=your-key-here")
        print()
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # --- Step 2: Create the Gemini API client ---
    client = genai.Client(api_key=api_key)

    # --- Step 3: Extract text from the PDF ---
    print(f"Reading PDF: {pdf_path}")
    paper_text = extract_pdf_text(pdf_path)
    print(f"Extracted {len(paper_text):,} characters from paper.\n")

    # Wrap the paper text in tags so the AI can clearly identify it
    paper_context = f"<paper>\n{paper_text}\n</paper>"

    # This list stores all dialogue parts for saving later
    dialogue_parts: list[dict] = []

    # --- Step 4: Moderator Opening ---
    # The moderator introduces the paper and sets up the debate
    print("=" * 70)
    print("MODERATOR — Opening")
    print("=" * 70)
    opening = call_agent(
        client,
        MODERATOR_SYSTEM,
        [{"role": "user", "content": f"{paper_context}\n\nWrite a brief opening for the dialogue about this paper."}],
        model,
    )
    print(opening)
    print()
    dialogue_parts.append({"role": "Moderator", "type": "opening", "text": opening})

    # --- Step 5: Initialize conversation histories ---
    # Each agent has its own conversation history so the AI
    # remembers what was said in previous rounds.

    # Skeptic starts with the paper and is asked for initial critique
    skeptic_history = [
        {"role": "user", "content": f"{paper_context}\n\nYou are about to discuss this paper with the Visionary. Start by giving your initial critical assessment — what are the biggest methodological concerns?"}
    ]

    # Visionary's history starts empty — it will be populated
    # after the Skeptic's first response
    visionary_history = []

    # --- Step 6: Run the dialogue rounds ---
    for round_num in range(1, rounds + 1):

        # === Skeptic's turn ===
        print("-" * 70)
        print(f"SKEPTIC — Round {round_num}")
        print("-" * 70)

        # Call the API with the Skeptic's personality and history
        skeptic_response = call_agent(client, SKEPTIC_SYSTEM, skeptic_history, model)
        print(skeptic_response)
        print()

        # Save the Skeptic's response
        dialogue_parts.append({"role": "Skeptic", "round": round_num, "text": skeptic_response})

        # Add the Skeptic's response to its own history
        # (role="model" because it was the AI's response)
        skeptic_history.append({"role": "model", "content": skeptic_response})

        # === Feed Skeptic's response to the Visionary ===
        # The Visionary needs to see what the Skeptic said so it can respond
        if round_num == 1:
            # First round: give the Visionary the full paper + Skeptic's opening
            visionary_history = [
                {"role": "user", "content": f"{paper_context}\n\nThe Skeptic opens with this critique:\n\n{skeptic_response}\n\nRespond with your perspective — where do you see the promise in this work, and how do you address their concerns?"}
            ]
        else:
            # Later rounds: just add the Skeptic's latest response
            visionary_history.append({"role": "user", "content": f"The Skeptic responds:\n\n{skeptic_response}"})

        # === Visionary's turn ===
        print("-" * 70)
        print(f"VISIONARY — Round {round_num}")
        print("-" * 70)

        # Call the API with the Visionary's personality and history
        visionary_response = call_agent(client, VISIONARY_SYSTEM, visionary_history, model)
        print(visionary_response)
        print()

        # Save the Visionary's response
        dialogue_parts.append({"role": "Visionary", "round": round_num, "text": visionary_response})

        # Add the Visionary's response to its own history
        visionary_history.append({"role": "model", "content": visionary_response})

        # === Feed Visionary's response back to the Skeptic ===
        # (only if there are more rounds to go)
        if round_num < rounds:
            skeptic_history.append({"role": "user", "content": f"The Visionary responds:\n\n{visionary_response}"})

    # --- Step 7: Moderator Closing Synthesis ---
    # Build a transcript of the entire dialogue for the moderator
    dialogue_transcript = ""
    for part in dialogue_parts:
        if part["role"] == "Moderator":
            continue  # Skip the moderator's own opening
        dialogue_transcript += f"\n[{part['role']} — Round {part.get('round', '')}]\n{part['text']}\n"

    print("=" * 70)
    print("MODERATOR — Closing Synthesis")
    print("=" * 70)

    # The moderator reads the full paper + full dialogue and writes a synthesis
    closing = call_agent(
        client,
        MODERATOR_SYSTEM,
        [{"role": "user", "content": f"{paper_context}\n\nHere is the full dialogue:\n{dialogue_transcript}\n\nWrite your closing synthesis."}],
        model,
    )
    print(closing)
    print()
    dialogue_parts.append({"role": "Moderator", "type": "closing", "text": closing})

    # --- Step 8: Save output to file (if requested) ---
    if output_file:
        save_dialogue(dialogue_parts, output_file)
        print(f"Dialogue saved to: {output_file}")

    return dialogue_parts


# ============================================================
# SAVE DIALOGUE TO FILE
# ============================================================

def save_dialogue(parts: list[dict], output_file: str):
    """
    Saves the dialogue to a file.

    Supports two formats:
        - .json: Structured data with role, round, and text fields.
        - .txt (or any other extension): Human-readable formatted text.

    Args:
        parts:       The list of dialogue parts from run_dialogue().
        output_file: Path to the output file.
    """
    ext = Path(output_file).suffix.lower()

    if ext == ".json":
        # JSON format — structured, easy to parse programmatically
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parts, f, indent=2, ensure_ascii=False)
    else:
        # Text format — human-readable with dividers
        with open(output_file, "w", encoding="utf-8") as f:
            for part in parts:
                role = part["role"]
                round_num = part.get("round", "")
                part_type = part.get("type", "")

                # Moderator sections get double-line dividers (=====)
                if role == "Moderator":
                    f.write(f"{'=' * 70}\n")
                    f.write(f"MODERATOR — {part_type.title()}\n")
                    f.write(f"{'=' * 70}\n\n")
                # Agent sections get single-line dividers (-----)
                else:
                    label = f"{role.upper()} — Round {round_num}"
                    f.write(f"{'-' * 70}\n")
                    f.write(f"{label}\n")
                    f.write(f"{'-' * 70}\n\n")

                # Write the actual dialogue text
                f.write(part["text"])
                f.write("\n\n")


# ============================================================
# CLI ENTRY POINT
# This runs when you execute: python dialogue.py paper.pdf
# ============================================================

def main():
    """
    Parses command-line arguments and starts the dialogue.

    Usage examples:
        python dialogue.py paper.pdf
        python dialogue.py paper.pdf --rounds 5
        python dialogue.py paper.pdf --model gemini-2.5-pro-preview-05-06
        python dialogue.py paper.pdf -o output.txt
        python dialogue.py paper.pdf -o output.json
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Generate a Skeptic vs. Visionary dialogue about an AI research paper."
    )

    # Required argument: the PDF file path
    parser.add_argument("pdf", help="Path to the research paper PDF")

    # Optional arguments
    parser.add_argument("-r", "--rounds", type=int, default=3,
                        help="Number of dialogue rounds (default: 3)")
    parser.add_argument("-m", "--model", default="gemini-2.0-flash",
                        help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument("-o", "--output",
                        help="Output file path (.txt or .json)")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Check that the PDF file actually exists
    if not Path(args.pdf).exists():
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)

    # Run the dialogue!
    run_dialogue(args.pdf, rounds=args.rounds, model=args.model, output_file=args.output)


# This ensures main() only runs when the script is executed directly,
# not when it's imported as a module by another script.
if __name__ == "__main__":
    main()
