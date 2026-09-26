# Baseline NumPy CPU demo sequence (meetup / live coding)
.PHONY: help verify export-gpu-weights demo demo-batch demo-plot demo-report demo-dashboard demo-open clean-bench

export-gpu-weights:
	$(PYTHON) tools/export_gpt2_weights.py
	$(PYTHON) tools/export_kernel_tests.py gpu-provision gpu-bootstrap

gpu-provision:
	bash scripts/runpod_provision_gpu.sh

gpu-bootstrap:
	bash scripts/runpod_remote_bootstrap.sh

PROMPT ?= The strangest thing I found inside my refrigerator was
PYTHON ?= .venv/bin/python
BENCH_DIR ?= bench/runs

help:
	@echo "Targets:"
	@echo "  make verify       HF parity gate (run before trusting demos)"
	@echo "  make demo         Full narrative batch -> $(BENCH_DIR)/"
	@echo "  make demo-batch   Same as demo"
	@echo "  make demo-plot       Matplotlib PNGs + demo.html + terminal table"
	@echo "  make demo-open       Regenerate demo.html and open in browser"
	@echo "  make demo-report     Prompt + completion + metrics -> $(BENCH_DIR)/report.txt"
	@echo "  make demo-dashboard  Same as demo-plot (HTML + graphs)"
	@echo "  make clean-bench     Remove $(BENCH_DIR)"
	@echo "  make gpu-provision   RunPod GPU + print SSH (needs RUNPOD_API_KEY)"
	@echo "  make gpu-bootstrap   Upload repo + cuda_oxide_host_setup (needs RUNPOD_SSH)"

verify:
	$(PYTHON) tools/verify_hf.py --strict

demo demo-batch: verify
	@mkdir -p $(BENCH_DIR)
	./scripts/demo_batch.sh "$(PROMPT)" $(BENCH_DIR)
	$(PYTHON) tools/summarize_bench.py $(BENCH_DIR)
	$(PYTHON) tools/report_bench.py $(BENCH_DIR) $(BENCH_DIR)/report.txt
	$(PYTHON) tools/render_demo.py $(BENCH_DIR)

demo-plot: demo-dashboard
	$(PYTHON) tools/summarize_bench.py $(BENCH_DIR)

demo-report:
	$(PYTHON) tools/report_bench.py $(BENCH_DIR) $(BENCH_DIR)/report.txt

demo-dashboard:
	$(PYTHON) tools/render_demo.py $(BENCH_DIR)

demo-open:
	$(PYTHON) tools/render_demo.py $(BENCH_DIR) --open

clean-bench:
	rm -rf $(BENCH_DIR)
