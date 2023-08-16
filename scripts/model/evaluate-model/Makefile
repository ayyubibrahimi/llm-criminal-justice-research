dfa_dir := ../../../data/convictions/evaluate/input
dfb_dir := ../../../data/convictions/evaluate/input
output_dir := ../../../data/convictions/evaluate/output
script := src.py
overall_summary := ../../../data/convictions/evaluate/output/overall-summary.csv

.PHONY: all clean

all:
	@echo "Processing files..."
	@for file in $(dfa_dir)/*-allqueries.csv; do \
		$(MAKE) process_allqueries FILE="$$file"; \
	done
	@for file in $(dfa_dir)/*-1query.csv; do \
		$(MAKE) process_1query FILE="$$file"; \
	done
	@echo "All files processed."
	@touch all_done

process_allqueries:
	@echo "Processing allqueries file: $(FILE)"
	@mkdir -p "$(output_dir)"
	@base_name=$$(basename "$(FILE)" -allqueries.csv); \
	python "$(script)" \
		--input_dfa="$(FILE)" \
		--input_dfb="$(dfb_dir)/$$base_name-groundtruth.csv" \
		--output="$(output_dir)/$$base_name-allqueries.csv"
	@touch "$(output_dir)/$$base_name-allqueries.done"

process_1query:
	@echo "Processing 1query file: $(FILE)"
	@mkdir -p "$(output_dir)"
	@base_name=$$(basename "$(FILE)" -1query.csv); \
	python "$(script)" \
		--input_dfa="$(FILE)" \
		--input_dfb="$(dfb_dir)/$$base_name-groundtruth.csv" \
		--output="$(output_dir)/$$base_name-1query.csv"
	@touch "$(output_dir)/$$base_name-1query.done"

clean:
	rm -rf "$(output_dir)"
	rm -f all_done
