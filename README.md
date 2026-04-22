# Paper Dialogue

Two AI agents debate a research paper:

- **Skeptic** — Finds flaws in methodology, statistical reasoning, and overclaimed results.
- **Visionary** — Identifies future applications, extensions, and potential impact.
- **Moderator** — Introduces the paper and synthesizes the debate.

Uses **Google Gemini API** (free tier — no payment required).

## How It Works

1. PDF is parsed and extracted as text.
2. Moderator introduces the paper.
3. For each round: Skeptic critiques → Visionary responds (each sees the other's prior messages).
4. Moderator writes a closing synthesis weighing both sides.

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/paper-dialogue.git
cd paper-dialogue

# Install dependencies
pip install -r requirements.txt

# Get your free Gemini API key at:
# https://aistudio.google.com/apikey

# Set your API key
export GEMINI_API_KEY=your-key-here
```

## Usage

```bash
# Basic run (3 rounds, Gemini Flash)
python dialogue.py paper.pdf

# More rounds
python dialogue.py paper.pdf --rounds 5

# Use a different model
python dialogue.py paper.pdf --model gemini-2.5-pro-preview-05-06

# Save output to a file
python dialogue.py paper.pdf -o output.txt
python dialogue.py paper.pdf -o output.json
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-r`, `--rounds` | Number of dialogue rounds | 3 |
| `-m`, `--model` | Gemini model to use | gemini-2.0-flash |
| `-o`, `--output` | Output file (.txt or .json) | None (prints to terminal) |

## Example Output

```
======================================================================
MODERATOR — Opening
======================================================================
Today we examine "Attention Is All You Need"...

----------------------------------------------------------------------
SKEPTIC — Round 1
----------------------------------------------------------------------
While the Transformer architecture is elegant, I have several concerns
about the experimental methodology...

----------------------------------------------------------------------
VISIONARY — Round 1
----------------------------------------------------------------------
The Skeptic raises fair points, but let me highlight why this work
fundamentally changes the trajectory of NLP...

======================================================================
MODERATOR — Closing Synthesis
======================================================================
After three rounds of debate, both agents found common ground on...
```

## Cost

**Free.** The Gemini API free tier is generous enough for regular use. No credit card required.

## License

MIT
