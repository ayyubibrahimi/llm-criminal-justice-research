txt_dir := ../../../data/convictions/evaluate-baseline/docs
csv_dir := ../../../data/convictions/evaluate-baseline/docs
script := baseline.py

.PHONY: all clean process_files

all:
	@echo "Processing Word files..."
	@for file in $(txt_dir)/*.docx; do \
		$(MAKE) process_files FILE="$$file"; \
	done
	@echo "All Word files processed."

process_files:
	@echo "Processing Word file: $(FILE)"
	@mkdir -p "$(csv_dir)"
	@base_name=$$(basename "$(FILE)" .docx); \
	python "$(script)" \
		--input="$(FILE)" \
		--output="$(csv_dir)/$$base_name.csv"

clean:
	rm -rf "$(csv_dir)"
