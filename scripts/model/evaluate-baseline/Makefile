dfa_dir := data/input
dfb_dir := data/groundtruth
output_dir := data/output
script := src.py

.PHONY: all clean process_files

all: process_files
	@echo "All files processed."

process_files:
	@find $(dfa_dir) -type f -name "*.csv" ! -name "*-groundtruth.csv" -print0 | \
	xargs -r -0 -I{} $(MAKE) process_csv FILE="{}"

process_csv:
	@echo "Processing file: $(FILE)"
	@mkdir -p "$(output_dir)"
	@base_name=$$(basename "$(FILE)" .csv); \
	python "$(script)" \
		--input_dfa="$(FILE)" \
		--input_dfb="$(dfb_dir)/$$base_name.docx-groundtruth.csv" \
		--output="$(output_dir)/$$base_name-comparison.csv"
	@touch "$(output_dir)/$$base_name-comparison.done"

clean:
	rm -rf "$(output_dir)"
	rm -f all_done
