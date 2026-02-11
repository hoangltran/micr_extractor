# MICRExtraction: Reading the Secret Language at the Bottom of Checks

## What This Project Is

Every check you've ever written has a line of strange-looking characters printed at the bottom in magnetic ink. That line — the MICR line — is the financial system's barcode. It tells machines who you bank with, what your account number is, and which check this is. It's been the backbone of check processing since the 1950s, and it still handles billions of transactions every year.

This project reads that line from an image and extracts the data — no internet connection, no cloud APIs, no sending your banking information to anyone's server. Just Python, OpenCV, and a surprisingly elegant trick involving a French mathematician from 1962.

---

## The Characters: E-13B, a Font With Only 14 Letters

MICR E-13B is the font used on checks in the US, Canada, UK, and Australia. It has exactly 14 characters:

- **Digits 0-9** — the numbers you'd expect
- **⑈ Transit** — surrounds the routing number (looks like a vertical bar with two dots)
- **⑆ On-Us** — separates account and check number fields (two thin vertical bars)
- **⑇ Amount** — surrounds the check amount (a thick bar with horizontal wings)
- **⑉ Dash** — a separator within fields (a horizontal bar)

A typical MICR line reads like this:

```
⑈267084131⑈  790319013⑆1024
 └─routing─┘  └─account──┘└check┘
```

The routing number `267084131` identifies the bank. The account number `790319013` is the customer's account. `1024` is the check number. The symbols are delimiters — they tell you where each field starts and ends.

Here's the clever part: the routing number has a **built-in error-detection checksum**. Multiply each digit by the weights `[3, 7, 1, 3, 7, 1, 3, 7, 1]`, sum them up, and the total must be divisible by 10. For `267084131`:

```
2×3 + 6×7 + 7×1 + 0×3 + 8×7 + 4×1 + 1×3 + 3×7 + 1×1
= 6 + 42 + 7 + 0 + 56 + 4 + 3 + 21 + 1 = 140
140 ÷ 10 = 14, remainder 0 ✓
```

This means we can **verify** that we read the routing number correctly. If our OCR makes even a single-digit mistake, the checksum will almost certainly fail. It's like having an answer key for a portion of the test.

---

## Why Not Just Use OCR?

This was the first question we had to answer. The obvious approach is: throw Tesseract or Google's OCR at it and call it a day. We evaluated four approaches:

| Approach | Digit Accuracy | Symbol Accuracy | Offline? | Dependencies |
|----------|---------------|-----------------|----------|--------------|
| Tesseract OCR | Good (85-95%) | Poor — misclassifies ⑈ ⑆ ⑇ ⑉ | Yes | Tesseract binary + trained data |
| PaddleOCR / EasyOCR | Good (85-90%) | None — not trained on MICR | Yes | PaddlePaddle (500MB+) or PyTorch |
| Template Matching (pixel correlation) | Poor — too brittle | Poor — scale-sensitive | Yes | OpenCV only |
| **Structural Feature Matching** | **Excellent (95%+)** | **Excellent** | **Yes** | **OpenCV only** |

Standard OCR engines are trained on general text. They handle digits reasonably well, but MICR special symbols are alien to them. Tesseract sees a Transit symbol (⑈) and guesses it's a bracket, a pipe character, or just gives up. PaddleOCR doesn't even try — it's never seen these characters in its training data.

Pixel-level template matching (sliding a reference image across the input) sounds promising, but it's fragile. A slight change in font rendering, DPI, or image quality breaks the correlation. We tried it early on and got 30% confidence scores. The templates we drew programmatically had sharp rectangular bars, but the actual MICR characters had softer edges and slightly different proportions. The pixel patterns just didn't match.

The winning approach was **structural feature matching** — instead of comparing raw pixels, we compare *what the character looks like* at a structural level: its shape moments, its density profile, its symmetry. This is robust to scale changes, minor distortions, and rendering differences. And the beautiful part: since MICR has only 14 characters, each one has a very distinctive structural fingerprint. Digit 1 (a single vertical bar) looks nothing like digit 8 (a full rectangular frame) at the feature level.

We kept Tesseract as an optional secondary engine for a "second opinion" on digits, but the structural matcher is the primary brain.

---

## The Technical Architecture

Here's how a check image becomes structured data:

```
                  ┌──────────────┐
                  │  Check Image │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │ Source Detect │ Is this from a scanner or a phone camera?
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼─────────┐ ┌────────▼─────────┐
     │ Scanner Pipeline  │ │ Camera Pipeline   │
     │ (light touch)     │ │ (heavy lifting)   │
     └────────┬──────────┘ └────────┬──────────┘
              │                     │
              └──────────┬──────────┘
                         │
                  ┌──────▼───────┐
                  │ MICR Locator │ Find the MICR line in the image
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼─────────┐ ┌────────▼─────────┐
     │ Template Engine   │ │ Tesseract Engine  │
     │ (primary)         │ │ (optional)        │
     └────────┬──────────┘ └────────┬──────────┘
              │                     │
              └──────────┬──────────┘
                         │
                  ┌──────▼───────┐
                  │  Consensus   │ Merge and arbitrate between engines
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │    Parser    │ ⑈ROUTING⑈ ACCOUNT⑆CHECK
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Validator   │ Routing checksum, field formats
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  MICRResult  │ Structured output with confidence
                  └──────────────┘
```

Each box is a separate module, testable in isolation. The pipeline is linear, which makes debugging straightforward — you can inspect the output at every stage.

---

## The Codebase: A Tour

```
MICRExtraction/
├── micr/                              # The main package
│   ├── __init__.py                    # Exports MICRExtractor and MICRResult
│   ├── api.py                         # The front door — MICRExtractor class
│   ├── cli.py                         # Command-line interface
│   ├── models.py                      # Data classes: MICRResult, CharacterResult
│   │
│   ├── preprocessing/                 # Image → clean binary
│   │   ├── common.py                  # Shared tools: grayscale, threshold, deskew
│   │   ├── scanner_pipeline.py        # Light preprocessing for scanner images
│   │   ├── camera_pipeline.py         # Heavy preprocessing for phone photos
│   │   └── micr_locator.py            # Find the MICR line in a full check image
│   │
│   ├── engines/                       # Character recognition
│   │   ├── base.py                    # Abstract interface (strategy pattern)
│   │   ├── template_engine.py         # PRIMARY: Hu moments + structural features
│   │   └── tesseract_engine.py        # SECONDARY: Tesseract OCR wrapper
│   │
│   ├── parsing/                       # Characters → structured data
│   │   ├── parser.py                  # Map symbols to routing/account/check
│   │   ├── validator.py               # ABA routing checksum
│   │   └── consensus.py               # Multi-engine result merging
│   │
│   └── resources/
│       └── templates/                 # 14 reference character images
│           ├── 0.png ... 9.png
│           ├── transit.png
│           ├── on_us.png
│           ├── amount.png
│           └── dash.png
│
├── scripts/
│   ├── generate_templates.py          # Draw templates programmatically
│   └── extract_templates_from_image.py # Extract templates from real images
│
├── tests/
│   └── test_extractor.py             # 16 tests covering the full pipeline
│
├── pyproject.toml                     # Dependencies and CLI entry point
├── MICR.png                           # Sample image 1
└── MICR2.png                          # Sample image 2
```

### How the pieces connect

**`api.py`** is the orchestrator. It instantiates the engines, calls the preprocessing pipeline, runs recognition, merges results, and returns a `MICRResult`. If you're importing this as a library, `MICRExtractor` is your entry point.

**`cli.py`** is a thin wrapper around `api.py` using `argparse`. It handles file I/O and output formatting (text or JSON). This separation means the core logic has zero dependency on CLI concerns.

**`models.py`** defines the data contracts. `CharacterResult` represents one recognized character with its confidence score and bounding box. `MICRResult` represents the fully parsed output. These are plain dataclasses — no behavior, just data. This makes them easy to serialize, test, and reason about.

**`preprocessing/`** turns a messy real-world image into a clean binary image where text is white and background is black. The scanner pipeline does minimal work (blur, Otsu threshold, deskew). The camera pipeline does the heavy lifting (bilateral filter for noise, CLAHE for uneven lighting, perspective correction for angled shots, adaptive thresholding).

**`engines/`** uses the **strategy pattern** — both engines implement the same `BaseMICREngine` interface with a `recognize()` method. This means you can swap engines, add new ones, or run them in parallel without changing any other code.

**`parsing/`** doesn't care how the characters were recognized. It just takes a list of `CharacterResult` objects and figures out the field structure based on symbol positions. This separation means the parser works identically whether the characters came from template matching, Tesseract, or a future neural network.

---

## The Recognition Engine: How We Actually Read Characters

This is the heart of the project, and it has three parts: **segmentation**, **feature extraction**, and **classification**.

### Part 1: Segmentation — Finding Individual Characters

Given a binary image of the MICR line, we need to find where each character is. OpenCV's `findContours` gives us the outline of every blob of white pixels. For most digits, that's one contour per character. Simple.

But the special symbols throw a wrench into this. The Transit symbol (⑈) isn't one connected shape — it's a vertical bar and two separate dots. That's **three** contours for one character. The On-Us symbol (⑆) is two separate thin bars. If we treat each contour as a separate character, we'll see three "characters" where there's actually one Transit symbol.

The solution is **contour grouping**, and it took three iterations to get right (more on the bugs below). We use a two-pass approach:

**Pass 1: Tight grouping.** Any contours that overlap or are extremely close (gap < 20% of character height, roughly 9 pixels) get merged. This handles the On-Us symbol's two bars (5px apart) and the Transit dots (which overlap in x-coordinate).

**Pass 2: Symbol completion.** We look for "short" groups — groups where no contour is taller than 75% of a typical digit. The Transit bar (height ~29px) and Transit dots (height ~14px) are both "short" compared to a full digit (~44px). If two adjacent short groups are within 30% of character height (~13px), they get merged. This joins the Transit bar with its dots.

The key insight: **full-height digits never get merged in Pass 2** because they're not "short." This is how we avoid accidentally merging a digit with a neighboring On-Us symbol even when the gap between them is the same as the gap inside a Transit symbol.

### Part 2: Feature Extraction — The Fingerprint

For each segmented character, we compute a feature vector — a numerical "fingerprint" that describes the character's structure. We extract 11 features:

1. **Hu Moments** (the star of the show) — Seven numbers derived from the character's shape that are invariant to scale, rotation, and translation. Named after Ming-Kuei Hu who described them in 1962. The first four moments are the most discriminative and capture the character's overall mass distribution.

2. **Density** — What fraction of the bounding box is filled with white pixels. Digit 7 (just two bars) has low density (~0.33). Digit 8 (a full frame) has high density (~0.56).

3. **Aspect Ratio** — Width divided by height. Digit 1 is narrow (~0.45). Digit 0 is wide (~0.80).

4. **Projection Profiles** — Sum the pixels in each row (horizontal profile) and each column (vertical profile), normalize, and resample to 16 bins. These capture the internal structure — digit 8 has three peaks in its horizontal profile (top bar, middle bar, bottom bar), while digit 1 has a flat profile.

5. **Connected Components** — How many separate blobs are in the character. Most digits have 1. Transit has 3.

6. **Symmetry** — How similar is the character to its mirror image. Digit 0 and 8 are very symmetric. Digit 7 is not.

7. **Half-Densities** — The density of each quadrant (top/bottom, left/right). Digit 2 has more pixels in the top-right and bottom-left. Digit 5 has the opposite pattern.

### Part 3: Classification — Finding the Best Match

We compare the input character's feature vector against the stored reference features for all 14 characters. Each feature contributes a similarity score, weighted by importance:

| Feature | Weight | Why This Weight |
|---------|--------|-----------------|
| Hu Moments | 3.0 (24%) | Most discriminative — captures overall shape |
| Density | 2.0 (16%) | Quick separator between open and closed forms |
| Aspect Ratio | 2.0 (16%) | Separates narrow (1, 3) from wide (0, 8) chars |
| Horizontal Profile | 1.5 (12%) | Internal horizontal structure |
| Vertical Profile | 1.5 (12%) | Internal vertical structure |
| Component Count | 1.0 (8%) | Separates simple digits from multi-part symbols |
| Half-Density | 1.0 (8%) | Quadrant distribution catches asymmetric chars |
| Symmetry | 0.5 (4%) | Useful tiebreaker, but least reliable |

The final score is the weighted average, producing a confidence between 0 and 1. On clean scanner images, most characters score above 0.95.

---

## The Consensus System: Two Heads Are Better Than One

When Tesseract is enabled, we have two sets of recognized characters. The consensus module merges them:

1. **Align by position** — Match characters from both engines based on overlapping bounding boxes (>30% overlap required).

2. **For symbols** — Always trust the template engine. Tesseract has no idea what ⑈ or ⑆ look like.

3. **For digits where both agree** — Boost confidence: `min(1.0, (primary_conf + secondary_conf) / 1.5)`. Agreement is strong evidence.

4. **For digits where they disagree** — Take the one with higher confidence, but apply a 20% penalty. Disagreement is a yellow flag.

This is conceptually similar to how ensemble methods work in machine learning — multiple independent classifiers combined produce better results than any single one.

---

## Technologies Used

### OpenCV (opencv-python-headless)
The workhorse. We use it for:
- Image I/O (`imread`)
- Color conversion (`cvtColor`)
- Blurring (`GaussianBlur`, `bilateralFilter`)
- Thresholding (`threshold` with Otsu, `adaptiveThreshold`)
- Contour detection (`findContours`)
- Morphological operations (`morphologyEx`)
- Moment computation (`moments`, `HuMoments`)
- Connected components (`connectedComponents`)
- Perspective transforms (`getPerspectiveTransform`, `warpPerspective`)
- CLAHE contrast enhancement (`createCLAHE`)

The `headless` variant doesn't include GUI components (no `imshow`), keeping the dependency footprint small for server deployments.

### NumPy
Array math. Projection profiles, correlation coefficients, feature vector comparisons. NumPy turns what would be nested Python loops into fast vectorized operations.

### Pillow
Image format support. While OpenCV handles the heavy processing, Pillow provides broader format compatibility.

### Tesseract + pytesseract (optional)
Google's open-source OCR engine. We use it in single-line mode (`--psm 7`) with LSTM engine (`--oem 1`) and a digit-only whitelist. It's an optional dependency — the template engine works without it.

### Why Python?
The ecosystem is unmatched for this task. OpenCV, NumPy, and Tesseract all have mature, well-documented Python bindings. The development speed is high, and for a POC processing individual images, Python's runtime performance is more than adequate. If throughput became critical (thousands of checks per second), the OpenCV calls are already running in C++ under the hood — Python is just orchestrating.

---

## Lessons Learned: The Bugs, The Fixes, and The Wisdom

### Bug #1: The Template That Didn't Look Like Anything

**What happened:** Our first approach was to generate E-13B character templates programmatically — draw rectangles on a canvas to approximate each character. We then used pixel-level correlation (Pearson coefficient) to match input characters against these templates. The result: 30% confidence, everything misidentified.

**Why it happened:** The programmatic templates had sharp rectangular bars at specific positions. The actual MICR characters in real images have softer edges, slightly different proportions, and rendering artifacts from printing/scanning. Pixel correlation is brutally sensitive to these differences. A template digit "2" and the real digit "2" looked nothing alike at the pixel level, even though a human would instantly recognize them as the same character.

**The fix:** We switched from pixel-level comparison to **structural feature matching** (Hu moments + density + projections). These features describe *what the character looks like* rather than *exactly which pixels are on*. A "2" has the same Hu moments whether it's drawn with sharp rectangles or with slightly rounded, anti-aliased strokes.

**The lesson:** When comparing things, ask yourself what level of abstraction is appropriate. Comparing pixels is like comparing two people by checking if every hair on their head is in the exact same position. Comparing structural features is like comparing their height, build, and facial proportions — much more robust and meaningful.

This is a pattern that comes up everywhere in engineering. Raw data comparison is fragile. Feature extraction creates a more meaningful representation.

### Bug #2: The Gap That Was Too Greedy

**What happened:** After fixing the matching, we had a new problem: the contour grouping was merging adjacent digits into single characters. The output showed 16 characters instead of 25, with many "characters" having bounding boxes 75+ pixels wide (normal digits are ~25-35px wide).

**Why it happened:** Our initial gap threshold was based on `median_character_width * 0.7`. This was about 17 pixels. But some inter-character gaps were only 12-13 pixels — smaller than our threshold. So adjacent digits got grouped together.

**The fix:** We switched the threshold to be based on character *height* rather than width: `median_height * 0.25 ≈ 11px`. This worked because the relationship between character height and the within-symbol gap is more consistent than the relationship between character width and the gap.

**The lesson:** When designing heuristics, anchor them to the most stable measurement available. Character height is remarkably consistent across a MICR line (all characters are the same height by specification). Character width varies a lot (digit 1 is half the width of digit 0). A threshold based on a stable anchor is more robust than one based on a variable one.

### Bug #3: The Gap That Was Exactly Wrong

**What happened:** After fixing the threshold, we had a maddening edge case. The Transit symbol's internal gap (bar to dots) was exactly 11px. The inter-character gap between digit 3 and the On-Us symbol was also exactly 11px. With a threshold of 11, using `< 11` meant Transit parts were NOT grouped (11 is not less than 11). Using `<= 11` would group them, but would also incorrectly merge digit 3 with On-Us.

**Why it happened:** Two fundamentally different gaps happened to be the same number of pixels. No single threshold could distinguish them.

**The fix:** We implemented a **two-pass grouping** algorithm:
- **Pass 1** uses a tight threshold (20% of character height ≈ 9px) that only groups truly overlapping or adjacent contours. This correctly groups the On-Us bars (5px apart) but leaves the Transit bar separate from its dots (11px apart).
- **Pass 2** specifically looks for "short" contour groups (shorter than 75% of a full character) and merges them with adjacent short groups within a broader distance (30% of height ≈ 13px). The Transit bar is short. The Transit dots are short. They get merged. But digit 3 is full-height, so it never triggers Pass 2 merging with the (short) On-Us symbol.

**The lesson:** When a single-pass algorithm can't handle all cases, consider whether the cases have some distinguishing property beyond the one you're currently using. Here, the distinguishing property was **contour height** — symbol parts are short, digits are tall. The two-pass approach uses this additional dimension to break the tie.

More generally: if you're trying to draw a line between two categories and find they overlap on one axis, look for another axis where they separate. This is basically what kernel methods do in machine learning, and it's a useful mental model for any classification problem.

### Bug #4: The Missing Digit 5

**What happened:** Testing on MICR2.png, every digit 5 was being recognized as digit 2. Three separate 5s, all misread with ~89% confidence.

**Why it happened:** Our reference templates were extracted from MICR.png, which happened to contain no digit 5s. The only "5" template we had was the programmatic one — drawn with rectangles, with completely wrong structural features. The real digit 5 and the real digit 2 happen to have similar aspect ratios and density, so without an accurate template, 2 was the closest match.

**The fix:** We extracted a real digit 5 template from MICR2.png. Confidence for 5s immediately jumped from ~89% (wrong) to ~96% (correct).

**The lesson:** Your system is only as good as its reference data. If your training set (or in our case, template set) has gaps, the system will silently fail on those gaps. It won't throw an error — it will confidently give you the wrong answer. This is arguably worse than crashing.

The mitigation: always test with diverse inputs. If MICR.png was our only test image, we'd never have found this bug. The check with all ten digits represented would be the ideal template source.

### Bug #5: The Phantom Transit Symbol

**What happened:** Also on MICR2.png, a digit 0 in the middle of the account number was being recognized as a Transit symbol (⑈). The routing number looked fine, but the account number was split in two by this phantom Transit.

**Why it happened:** A tiny noise speck (9×11 pixels) was sitting just above the digit 0. It was small enough to look like nothing, but it passed our size filter (minimum height was 10% of image height = 9.5px, and 11 > 9.5). Because its x-coordinate overlapped with the digit 0, they got grouped together, creating a character region that was much taller (72px instead of the normal 53px) and had an unusual shape. The feature matcher thought this weird, elongated shape looked most like a Transit symbol.

**The fix:** We tightened the noise filter from `0.10 × image_height` to `0.13 × image_height` for minimum contour height, and raised the minimum contour area from 20 to 50 pixels. The noise speck (area ≈ 99px²... actually still passes 50, but at 0.13 × 95 = 12.35, the speck's height of 11 now fails the filter). The Transit dots (height ~17px) safely pass both filters.

**The lesson:** Noise filtering thresholds need to be set with knowledge of both what you want to keep and what you want to reject. List out the smallest legitimate contour (Transit dots: ~17px tall, ~200px² area) and the largest noise you've observed (~11px tall, ~100px² area), then set your threshold between them. Leave margin on both sides.

More broadly: a single bad pixel can poison everything downstream. Preprocessing isn't glamorous, but it's where most real-world vision systems succeed or fail.

---

## How Good Engineers Think: Patterns From This Project

### 1. Start With the Constraint, Not the Solution

We didn't start by asking "what OCR library should we use?" We started by asking "what are the constraints?" Offline-only. Enterprise reliability. Camera AND scanner inputs. Only then did we evaluate solutions against those constraints. This prevented us from falling in love with a tool that didn't fit.

### 2. Exploit Domain Knowledge

MICR E-13B has only 14 characters. That's an extraordinary constraint — most OCR problems deal with hundreds or thousands of character classes. This single fact made template matching viable where it wouldn't be for general text. The routing number checksum gave us a built-in verification mechanism. The fixed character pitch gave us a basis for contour grouping. Every piece of domain knowledge we used reduced the problem's difficulty.

When you're solving a problem in a specific domain, spend time learning that domain. The solution often comes from domain knowledge, not from a fancier algorithm.

### 3. Build in Layers, Test Each Layer

The pipeline is strictly linear: preprocess → locate → segment → recognize → parse → validate. Each stage has a well-defined input and output. When something went wrong, we could inspect the output of each stage and pinpoint exactly where the problem was. "Is the binary image clean? Yes. Are the contours correct? Yes. Are they grouped correctly? No." Debug the grouping.

If we'd built this as one monolithic function, debugging would have been detective work. Layers make it forensics.

### 4. Make the Common Case Fast, Handle Edge Cases Separately

The two-pass contour grouping is a perfect example. Pass 1 handles 90% of cases with a simple threshold. Pass 2 handles the remaining 10% (multi-part symbols) with a more nuanced rule. We didn't make Pass 1 more complex to handle symbols — we added a separate, targeted pass.

This is the [80/20 rule](https://en.wikipedia.org/wiki/Pareto_principle) applied to algorithm design. A simple solution handles most cases. A targeted fix handles the rest. Trying to build one elegant solution that handles everything usually produces something that handles nothing well.

### 5. Real Data Beats Synthetic Data

Our programmatic templates (rectangles on a canvas) had the right general structure but wrong fine details. Templates extracted from a real MICR image matched almost perfectly. Synthetic data is useful for bootstrapping, but it should be replaced with real-world data as soon as possible.

### 6. The Second Test Image Catches the First Image's Blind Spots

MICR.png had digits 0-4, 6-9, and symbols Transit and On-Us. It did NOT have digit 5, Amount, or Dash. Everything worked perfectly on MICR.png and was silently wrong on MICR2.png (which had 5s). The first test image gives you confidence. The second test image gives you truth.

---

## Potential Pitfalls and How to Avoid Them

### Pitfall: Over-relying on a single test image
MICR.png was our development image. It's tempting to tune everything to perfection on one image and declare victory. But the digit-5 bug only appeared on a second image. **Always test on at least 3-5 diverse images** covering different banks, check formats, and image qualities.

### Pitfall: Magic numbers in thresholds
The code has numbers like `0.13`, `0.20`, `0.75`, `0.30`. Each of these was derived from analyzing the actual pixel measurements of real MICR characters. If you ever need to adjust them, the comments explain the reasoning (e.g., "Transit dots are ~0.16x height, so 0.13 filters noise while keeping dots"). Without these comments, future maintainers would be afraid to touch them.

### Pitfall: Assuming binary images are truly binary
After Otsu thresholding, you'd expect every pixel to be 0 or 255. But if you apply a rotation (deskew) with cubic interpolation, the resampling creates intermediate values along edges. The template engine re-thresholds at 127 to clean this up. It's a subtle gotcha that can cause contour detection to behave unexpectedly.

### Pitfall: Template coverage gaps
If your template set doesn't include a character, the system will confidently match it to the closest character it does know. There's no "unknown" category. For production, ensure you have high-quality templates for all 14 characters, ideally extracted from multiple real-world images to capture font variations.

### Pitfall: Confusing pixel gaps with visual gaps
When analyzing contour gaps, the "gap" is measured from the right edge of one bounding box to the left edge of the next. But bounding boxes can be slightly larger than the visible character due to anti-aliasing. A "gap" of 11 pixels might visually look like 13 pixels of whitespace. Always measure from the actual bounding rectangles, not from visual inspection.

---

## Running the Project

### Install

```bash
pip install opencv-python-headless numpy Pillow

# Optional: for Tesseract secondary engine
pip install pytesseract
brew install tesseract    # macOS
# apt-get install tesseract-ocr  # Linux
```

### Command Line

```bash
# Basic extraction
python -m micr.cli check.png

# JSON output
python -m micr.cli check.png --format json

# With per-character details
python -m micr.cli check.png --verbose

# With Tesseract as secondary engine
python -m micr.cli check.png --tesseract
```

### Python API

```python
from micr import MICRExtractor

extractor = MICRExtractor()
result = extractor.extract("check.png")

print(result.routing_number)    # "267084131"
print(result.account_number)    # "790319013"
print(result.check_number)      # "1024"
print(result.routing_valid)     # True
print(result.overall_confidence) # 0.9687

# Process a numpy array directly
import cv2
image = cv2.imread("check.png")
result = extractor.extract_from_array(image)

# Get JSON-serializable output
data = result.to_dict()
```

### Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## What's Next: Paths for Improvement

1. **More reference templates.** Collect MICR images from 10-20 different banks and extract templates. Average the features across samples to create more robust reference vectors.

2. **A small CNN classifier.** With the current segmentation pipeline producing clean character ROIs, a tiny convolutional neural network (even 3-4 layers) trained on the 14 E-13B characters could replace the feature-matching classifier. The training set can be synthetically augmented from real templates.

3. **Batch processing.** Add `multiprocessing` support for processing directories of check images in parallel.

4. **REST API wrapper.** Wrap the Python API in a FastAPI service for integration with other enterprise systems, while keeping all processing local.

5. **CMC-7 support.** The other MICR standard, used in parts of Europe and Latin America, has a different character set but the same architectural approach would apply.

---

## Final Thought

This project started as "read some text from an image" and turned into a lesson in how domain knowledge, thoughtful preprocessing, and robust feature engineering can outperform brute-force approaches. The MICR E-13B standard was designed in the 1950s to be machine-readable — but the machines they had in mind were magnetic readers, not cameras. Sixty years later, a few hundred lines of Python and some 1962 mathematics can do it from a photograph. That's progress.
