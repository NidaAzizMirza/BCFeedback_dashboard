#!/bin/bash
# cleanup_project.sh
# ── Reorganizes the project root ───────────────────────────────────────────
# Safe to run while a pipeline job is in progress — this script never
# touches data/, models/, logs/, outputs/, or inputs/. It only moves loose
# files at the project root that look like earlier one-off exploration,
# fixes the .github/workflows/ location, and fixes a requirements filename
# mismatch. Nothing is deleted except one Excel temp lock file.
#
# Run from your project root:
#   bash cleanup_project.sh
#
# Review the "NEEDS YOUR INPUT" section at the bottom before deciding on
# svm_model.pkl / svm_label_encoder.pkl — this script does NOT move those.

set -e  # stop on first error

echo "=== 1. Fixing GitHub Actions workflow location ==="
mkdir -p .github/workflows
if [ -f "daily_pipeline.yml" ]; then
    mv daily_pipeline.yml .github/workflows/daily_pipeline.yml
    echo "  Moved daily_pipeline.yml -> .github/workflows/"
fi
if [ -f "weekly_pipeline.yml" ]; then
    mv weekly_pipeline.yml .github/workflows/weekly_pipeline.yml
    echo "  Moved weekly_pipeline.yml -> .github/workflows/"
fi

echo ""
echo "=== 2. Fixing requirements filename mismatch ==="
if [ -f "requirements_daily.txt" ] && [ ! -f "requirements-daily.txt" ]; then
    mv requirements_daily.txt requirements-daily.txt
    echo "  Renamed requirements_daily.txt -> requirements-daily.txt (matches workflow reference)"
fi

echo ""
echo "=== 3. Archiving old exploration scripts + their outputs ==="
mkdir -p archive/scripts archive/outputs archive/models

# Old one-off pipeline scripts (superseded by run_pipeline.py / update_raw_metrics.py)
for f in absapipeline.py entity_extraction.py summarisation_pipeline.py \
         sentiment.py compare.py playground.py playground2.py test.py; do
    [ -f "$f" ] && mv "$f" archive/scripts/ && echo "  Archived $f"
done

# Old exploration outputs (xlsx/png/csv from earlier experimentation)
for f in absa_summary.xlsx absa_sentiment_charts.png entity_analysis.xlsx \
         summaries.xlsx sentiment_comparison.xlsx svm_confusion_matrix.png \
         svm_results.xlsx svm_keyword_comparison.xlsx zeroshot_results.xlsx \
         zeroshot_validation.xlsx topic_results.xlsx topic_summary.xlsx \
         validation_results.xlsx comparison_output.xlsx final_results.xlsx \
         final_results_absa.xlsx final_results_entities.xlsx \
         final_results_multilabel.xlsx final_sentiment_analysis.xlsx \
         granular_results.xlsx cleaned_feedback.xlsx; do
    [ -f "$f" ] && mv "$f" archive/outputs/ && echo "  Archived $f"
done

# Excel temp lock file (safe to delete outright — recreated automatically)
[ -f "~\$final_sentiment_analysis.xlsx" ] && rm -f "~\$final_sentiment_analysis.xlsx" && echo "  Deleted Excel lock file"

# Old granular SVM model files (separate from the production svm_final_* models)
for f in svm_granular_model.pkl svm_granular_label_encoder.pkl; do
    [ -f "$f" ] && mv "$f" archive/models/ && echo "  Archived $f"
done

# Old topic model folder, if present
[ -d "review_topic_model" ] && mv review_topic_model archive/models/ && echo "  Archived review_topic_model/"

echo ""
echo "=== 4. Moving diagnostic scripts into their own folder ==="
mkdir -p diagnostics
for f in check_dupes.py diagnose_rating.py diagnose_master_dupes.py; do
    [ -f "$f" ] && mv "$f" diagnostics/ && echo "  Moved $f -> diagnostics/"
done

echo ""
echo "=== 5. Checking manuallytagged_complete.csv location ==="
if [ -f "manuallytagged_complete.csv" ]; then
    echo "  Found manuallytagged_complete.csv at root."
    echo "  pipeline_config.py expects it at data/manuallytagged_complete.csv"
    echo "  NOT moved automatically — confirm this is the right file, then run:"
    echo "    mv manuallytagged_complete.csv data/manuallytagged_complete.csv"
fi
if [ -f "manually_tagged.csv" ]; then
    echo "  Found manually_tagged.csv (different name, not referenced by pipeline_config.py)"
    echo "  Leaving in place — archive manually if it's no longer needed:"
    echo "    mv manually_tagged.csv archive/outputs/"
fi

echo ""
echo "=== Done ==="
echo ""
echo "=== NEEDS YOUR INPUT (not touched by this script) ==="
echo "svm_model.pkl and svm_label_encoder.pkl are sitting at the project root."
echo "pipeline_config.py expects models/svm_final_model.pkl and"
echo "models/svm_final_label_encoder.pkl instead."
echo ""
echo "Check what's inside models/ first:"
echo "    ls -la models/"
echo ""
echo "If models/svm_final_model.pkl already exists and works, the root-level"
echo "svm_model.pkl / svm_label_encoder.pkl are almost certainly old copies —"
echo "archive them with:"
echo "    mv svm_model.pkl svm_label_encoder.pkl archive/models/"
echo ""
echo "If models/ is missing those files, do NOT archive the root-level ones —"
echo "come back and let me know, since that would mean the pipeline is"
echo "currently relying on them."