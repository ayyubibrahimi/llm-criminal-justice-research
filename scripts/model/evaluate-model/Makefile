dfa_reports_dir := ../../../data/wrongful-convictions/json/output-4-og-params/reports
dfb_reports_dir := ../../../data/wrongful-convictions/json/output-4-og-params/reports

dfa_transcripts_dir := ../../../data/wrongful-convictions/json/output-4-og-params/transcripts
dfb_transcripts_dir := ../../../data/wrongful-convictions/json/output-4-og-params/transcripts

output_reports_dir := ../../../data/wrongful-convictions/json/evaluate/output-4-og-params/reports
output_transcripts_dir := ../../../data/wrongful-convictions/json/evaluate/output-4-og-params/transcripts


script := src.py

.PHONY: all clean

all: 
	$(MAKE) process_files_type TYPE=reports
	$(MAKE) process_files_type TYPE=transcripts

process_files_type:
	@echo "Processing files of type: $(TYPE)"
	@for file in $(dfa_$(TYPE)_dir)/*_six_queries.csv; do \
		$(MAKE) process_six_queries FILE="$$file" TYPE=$(TYPE); \
	done
	@for file in $(dfa_$(TYPE)_dir)/*_one_query.csv; do \
		$(MAKE) process_one_query FILE="$$file" TYPE=$(TYPE); \
	done
	@echo "All $(TYPE) files processed."
	@touch all_$(TYPE)_done

process_six_queries:
	@echo "Processing six queries file: $(FILE)"
	@mkdir -p "$(output_$(TYPE)_dir)"
	@base_name=$$(basename "$(FILE)" _six_queries.csv); \
	python "$(script)" \
		--input_dfa="$(FILE)" \
		--input_dfb="$(dfb_$(TYPE)_dir)/$$base_name-groundtruth.csv" \
		--output="$(output_$(TYPE)_dir)/$$base_name_six_queries.csv" \
		--file-type="$(TYPE)" 
	@touch "$(output_$(TYPE)_dir)/$$base_name_six_queries.done"

process_one_query:
	@echo "Processing _one_query file: $(FILE)"
	@mkdir -p "$(output_$(TYPE)_dir)"
	@base_name=$$(basename "$(FILE)" _one_query.csv); \
	python "$(script)" \
		--input_dfa="$(FILE)" \
		--input_dfb="$(dfb_$(TYPE)_dir)/$$base_name-groundtruth.csv" \
		--output="$(output_$(TYPE)_dir)/$$base_name_one_query.csv" \
		--file-type="$(TYPE)" 
	@touch "$(output_$(TYPE)_dir)/$$base_name_one_query.done"

clean:
	rm -rf "$(output_reports_dir)"
	rm -rf "$(output_transcripts_dir)"
	rm -f all_reports_done
	rm -f all_transcripts_done
