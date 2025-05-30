# FlashExamOMR

FlashExamOMR is a Python package for processing and analyzing Optical Mark Recognition (OMR) exam sheets. It provides tools to configure grid layouts for OMR sheets and to extract, match, and report exam results using scanned images and CSV mappings.

## Components

### 1. OMR-reader.py

This is the main script for processing scanned OMR exam sheets. It reads images, matches them to student names using a CSV mapping, extracts answers, and generates results and reports.

#### Key Features
- Processes a folder of scanned exam images.
- Uses a grid configuration to locate answer bubbles.
- Matches images to student names via a CSV file.
- Outputs results in CSV and JSON formats, and can generate PDF reports.

### 2. grid_setup_multi.py

This script is used to interactively configure the grid layout for your OMR sheets. It helps you define the positions of answer bubbles and other relevant fields, saving the configuration to a JSON file.

---

## Inputs

### 1. Exam Images
- **Location:** Typically in a subfolder under `inputs/exams/` (e.g., `inputs/exams/temaA/`).
- **Format:** PNG or JPG images of scanned OMR sheets.

### 2. grid_config.json
- **Purpose:** Defines the layout of the OMR sheet (positions of answer bubbles, ID fields, etc.).
- **How to create:** Use `grid_setup_multi.py` to interactively set up and save this file.

#### Example `grid_config.json`

```json
{
  "bubble_radius": 12,
  "questions": [
    {"x": 100, "y": 200, "choices": 4},
    {"x": 100, "y": 250, "choices": 4}
  ],
  "id_boxes": [
    {"x": 50, "y": 100, "width": 30, "height": 20}
  ],
  "page_size": [2480, 3508]
}
```

### 3. image-to-name.csv
- **Purpose:** Maps image filenames to student names or IDs.
- **Format:** CSV with at least two columns: `image_filename`, `student_name`.

### 4. answers.json / scoring.json (optional)
- **Purpose:** Define correct answers and scoring rules for automatic grading.

---

## Outputs

All outputs are written to the specified output directory (e.g., `output/temaA/`).

### 1. image-to-name.csv
- The mapping used for the current run, copied or generated as needed.

### 2. results.csv
- Contains the extracted answers for each student.

### 3. grades.csv
- Contains the calculated grades for each student (if scoring is enabled).

### 4. grades_with_names.csv
- Like `grades.csv`, but includes student names.

### 5. exam_report.pdf
- A PDF report summarizing the results (if enabled).

### 6. detections/
- Contains images or data showing detected bubbles and fields for debugging.

### 7. students-info/
- May contain per-student information or extracted data.

---

## Usage

### 1. Configure the Grid

Run the grid setup tool to create or adjust your grid configuration:

```bash
python grid_setup_multi.py --input example_sheet.png --output grid_config.json
```

### 2. Process Exam Sheets

Run the OMR reader to process a folder of scanned exams:

```bash
python OMR-reader.py \
  --input_folder inputs/exams/temaA \
  --output output/temaA \
  --csv inputs/image-to-name-temaA.csv \
  --grid_config grid_config.json \
  --answers_csv inputs/answers-temaA.json \
  --scoring_json inputs/scoring.json
```

#### Optional: Use a Pre-existing image-to-name.csv

```bash
python OMR-reader.py \
  ...other arguments... \
  --image-to-name-csv inputs/image-to-name-temaA.csv
```

---

## Notes

- Ensure your scanned images are clear and match the grid configuration.
- The `grid_config.json` must accurately reflect the layout of your OMR sheets.
- The mapping CSV must include all image filenames to be processed.

For further customization or troubleshooting, refer to the comments and docstrings in the code.
