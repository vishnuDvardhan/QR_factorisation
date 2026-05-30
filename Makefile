# =============================================================================
# QR factorization on Hexagon DSP (QuRT + HVX) — Makefile
#
# Sources:
#   qr_colmajor_mt.c     - unblocked column-major Householder QR (baseline)
#   qr_wy_blocked_mt.c   - WY-blocked QR (includes the baseline inline for A/B)
#
# Usage:
#   make                   # build both binaries
#   make blocked           # build qr_wy_blocked_mt only
#   make baseline          # build qr_colmajor_mt only
#   make sim               # run WY-blocked binary on the Hexagon simulator
#   make sim-baseline      # run unblocked baseline on the simulator
#   make clean             # wipe build/
#   make help              # show targets
#
# The simulator reads inparams.txt from one level above the binary's cwd
# (the C code does fopen("./../inparams.txt"), and --usefs maps build/hexagon
#  as the simulated FS).  So inparams.txt must live in build/.
# A default is auto-created on first run if missing.
# =============================================================================

# ---------------- SDK / toolchain ----------------
ifeq ($(HEXAGON_SDK_ROOT),)
    HEXAGON_SDK_ROOT ?= /home/vishnu/Qualcomm/Hexagon_SDK/6.5.0.0
endif

HEX_VERSION ?= v69

HEXAGON_TOOLS_ROOT ?= $(firstword $(sort $(wildcard $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/*)))
ifeq ($(HEXAGON_TOOLS_ROOT),)
HEXAGON_TOOLS_ROOT := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/19.0.07
endif

CC_HEXAGON  ?= hexagon-clang
HEXAGON_SIM ?= hexagon-sim

QURT_DIR   = $(HEXAGON_SDK_ROOT)/rtos/qurt/compute$(HEX_VERSION)
QURT_LIB   = $(QURT_DIR)/lib
TARGET_DIR = $(HEXAGON_TOOLS_ROOT)/Tools/target/hexagon/lib/$(HEX_VERSION)/G0
QCC_DIR    = $(HEXAGON_TOOLS_ROOT)/Tools/target/hexagon/lib/$(HEX_VERSION)/G0

# ---------------- Compile / link flags ----------------
CFLAGS_HEXAGON  = -O2 -m$(HEX_VERSION) -g
CFLAGS_HEXAGON += -Wall -Wextra -Wno-unused-parameter
CFLAGS_HEXAGON += -ffast-math
CFLAGS_HEXAGON += -I$(QURT_DIR)/include
CFLAGS_HEXAGON += -I$(QURT_DIR)/include/qurt
CFLAGS_HEXAGON += -I$(QURT_DIR)/include/posix
CFLAGS_HEXAGON += -Iinclude

# HVX flags (added on top of the base CFLAGS for vectorized sources)
CFLAGS_HVX = $(CFLAGS_HEXAGON) -mhvx -mhvx-length=128B -mhvx-qfloat

LDFLAGS_HEXAGON  = -m$(HEX_VERSION) -g
LDFLAGS_HEXAGON += -nodefaultlibs -nostdlib
LDFLAGS_HEXAGON += -Wl,--section-start -Wl,.interp=0x23000000
LDFLAGS_HEXAGON += -Wl,--dynamic-linker= -Wl,--force-dynamic
LDFLAGS_HEXAGON += -Wl,-E -Wl,-z -Wl,muldefs
LDFLAGS_HEXAGON += -Wl,--whole-archive

# QuRT link order (do not rearrange)
QURT_INIT_LIBS  = $(TARGET_DIR)/init.o
QURT_LINK_LIBS  = $(QURT_LIB)/crt1.o
QURT_LINK_LIBS += $(QURT_LIB)/debugmon.o
QURT_LINK_LIBS += $(QURT_LIB)/libqurt.a
QURT_LINK_LIBS += $(TARGET_DIR)/libc.a
QURT_LINK_LIBS += $(QCC_DIR)/libqcc.a
QURT_LINK_LIBS += $(TARGET_DIR)/libhexagon.a
QURT_LINK_LIBS += $(QURT_LIB)/libqurtcfs.a
QURT_LINK_LIBS += $(QURT_LIB)/libtimer_island.a
QURT_LINK_LIBS += $(QURT_LIB)/libtimer_main.a
QURT_LINK_LIBS += $(QURT_LIB)/libposix.a
QURT_FINI_LIBS  = $(TARGET_DIR)/fini.o

# ---------------- Sources / outputs ----------------
BASELINE_SRC    = qr_colmajor_mt.c
BLOCKED_SRC     = qr_wy_blocked_mt.c
ROWMAJOR_SRC    = qr_rowmajor_mt_block_alloc.c
ROWMAJOR_WY_SRC = qr_wy_blocked_rm.c
COMPARE_SRC     = qr_compare_all.c
POOL_SRC        = qr_pool_hh_unblocked.c
POOL_WY_SRC     = qr_pool_hh_wy_blocked.c
POOL_WY_GEMM_SRC = qr_pool_hh_wy_gemm.c

BUILD_DIR    = build
HEXAGON_DIR  = $(BUILD_DIR)/hexagon

BASELINE_TARGET    = $(HEXAGON_DIR)/qr_colmajor_mt
BLOCKED_TARGET     = $(HEXAGON_DIR)/qr_wy_blocked_mt
ROWMAJOR_TARGET    = $(HEXAGON_DIR)/qr_rowmajor_mt_block_alloc
ROWMAJOR_WY_TARGET = $(HEXAGON_DIR)/qr_wy_blocked_rm
COMPARE_TARGET     = $(HEXAGON_DIR)/qr_compare_all
POOL_TARGET        = $(HEXAGON_DIR)/qr_pool_hh_unblocked
POOL_WY_TARGET     = $(HEXAGON_DIR)/qr_pool_hh_wy_blocked
POOL_WY_GEMM_TARGET = $(HEXAGON_DIR)/qr_pool_hh_wy_gemm

# ---------------- Simulator config files ----------------
QURT_OS       = $(QURT_DIR)/sdksim_bin/runelf.pbn
QURT_MODEL    = $(QURT_DIR)/debugger/lnx64/qurt_model.so
HEXAGON_ISS_DIR = $(HEXAGON_TOOLS_ROOT)/Tools/lib/iss
OSAM_CFG      = $(HEXAGON_DIR)/osam.cfg
Q6SS_CFG      = $(HEXAGON_DIR)/q6ss.cfg

# inparams.txt lives in build/ (one level above the binary, per fopen path)
INPARAMS_FILE = $(BUILD_DIR)/inparams.txt

# Default inparams: m=256 n=128 nt=4 chunk=4 static=0 nb=16
# (m must be a multiple of 32 for HVX vector loads)
INPARAMS_DEFAULT := 256 128 4 4 0 16

# =============================================================================
# Phony targets
# =============================================================================
.PHONY: all baseline blocked rowmajor rowmajor-wy compare pool pool-wy pool-wy-gemm \
        sim sim-baseline sim-blocked sim-rowmajor sim-rowmajor-wy sim-compare sim-pool sim-pool-wy sim-pool-wy-gemm \
        clean help inparams cfgs

all: baseline blocked rowmajor rowmajor-wy compare pool pool-wy pool-wy-gemm

# ---------------- Build directory ----------------
$(HEXAGON_DIR):
	mkdir -p $(HEXAGON_DIR)

# ---------------- Default inparams.txt ----------------
# Create one only if it doesn't already exist; never overwrite a user file.
$(INPARAMS_FILE): | $(HEXAGON_DIR)
	@if [ ! -f $(INPARAMS_FILE) ]; then \
		echo "$(INPARAMS_DEFAULT)" > $(INPARAMS_FILE); \
		echo "Created default $(INPARAMS_FILE): $(INPARAMS_DEFAULT)"; \
		echo "  (format: m n nt chunk_size use_static nb)"; \
	fi

inparams: $(INPARAMS_FILE)
	@cat $(INPARAMS_FILE)

# ---------------- Simulator config files ----------------
$(OSAM_CFG): | $(HEXAGON_DIR)
	@echo "$(QURT_MODEL)" > $(OSAM_CFG)
	@echo "Created $(OSAM_CFG)"

$(Q6SS_CFG): | $(HEXAGON_DIR)
	@echo "$(HEXAGON_ISS_DIR)/qtimer.so --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1" > $(Q6SS_CFG)
	@echo "$(HEXAGON_ISS_DIR)/l2vic.so 32 0xFC910000" >> $(Q6SS_CFG)
	@echo "Created $(Q6SS_CFG)"

cfgs: $(OSAM_CFG) $(Q6SS_CFG)

# ---------------- Build: baseline (unblocked) ----------------
baseline: $(BASELINE_TARGET)

$(BASELINE_TARGET): $(BASELINE_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_colmajor_mt.o $(BASELINE_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_colmajor_mt.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built baseline: $@"

# ---------------- Build: blocked (WY, column-major) ----------------
blocked: $(BLOCKED_TARGET)

$(BLOCKED_TARGET): $(BLOCKED_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_wy_blocked_mt.o $(BLOCKED_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_wy_blocked_mt.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built blocked:  $@"

# ---------------- Build: row-major baseline ----------------
rowmajor: $(ROWMAJOR_TARGET)

$(ROWMAJOR_TARGET): $(ROWMAJOR_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_rowmajor_mt_block_alloc.o $(ROWMAJOR_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_rowmajor_mt_block_alloc.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built rowmajor: $@"

# ---------------- Build: qr_compare_all (all algorithms in one binary) ----------------
compare: $(COMPARE_TARGET)

$(COMPARE_TARGET): $(COMPARE_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_compare_all.o $(COMPARE_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_compare_all.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built compare:  $@"

# ---------------- Build: row-major WY-blocked ----------------
rowmajor-wy: $(ROWMAJOR_WY_TARGET)

$(ROWMAJOR_WY_TARGET): $(ROWMAJOR_WY_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_wy_blocked_rm.o $(ROWMAJOR_WY_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_wy_blocked_rm.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built rowmajor-wy: $@"

# ---------------- Run on simulator ----------------
# Common sim invocation — usage: $(call run-sim,<binary>)
define run-sim
	@command -v $(HEXAGON_SIM) > /dev/null 2>&1 || { \
		echo "ERROR: $(HEXAGON_SIM) not found in PATH."; exit 1; }
	@echo "----- inparams.txt -----"
	@cat $(INPARAMS_FILE)
	@echo "------------------------"
	$(HEXAGON_SIM) -m$(HEX_VERSION) --simulated_returnval --timing \
		--usefs $(HEXAGON_DIR) \
		--pmu_statsfile $(HEXAGON_DIR)/pmu_stats.txt \
		--cosim_file $(Q6SS_CFG) \
		--l2tcm_base 0xd800 \
		--subsystem_base 0xFC90 \
		--rtos $(OSAM_CFG) $(QURT_OS) -- \
		$(1)
endef

sim: sim-blocked

sim-blocked: blocked cfgs $(INPARAMS_FILE)
	@echo "Running WY-blocked QR (column-major) on Hexagon simulator..."
	$(call run-sim,$(BLOCKED_TARGET))

sim-baseline: baseline cfgs $(INPARAMS_FILE)
	@echo "Running unblocked baseline QR (column-major) on Hexagon simulator..."
	$(call run-sim,$(BASELINE_TARGET))

sim-compare: compare cfgs $(INPARAMS_FILE)
	@echo "Running qr_compare_all (all algorithms) on Hexagon simulator..."
	$(call run-sim,$(COMPARE_TARGET))

# ---------------- Build: pool variant (HH unblocked + persistent thread pool) ----------------
pool: $(POOL_TARGET)

$(POOL_TARGET): $(POOL_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_pool_hh_unblocked.o $(POOL_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_pool_hh_unblocked.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built pool:     $@"

sim-pool: pool cfgs $(INPARAMS_FILE)
	@echo "Running pool variant (HH unblocked + persistent thread pool)..."
	$(call run-sim,$(POOL_TARGET))

pool-wy: $(POOL_WY_TARGET)

$(POOL_WY_TARGET): $(POOL_WY_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_pool_hh_wy_blocked.o $(POOL_WY_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_pool_hh_wy_blocked.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built pool-wy:  $@"

sim-pool-wy: pool-wy cfgs $(INPARAMS_FILE)
	@echo "Running pool-wy variant (WY-blocked + persistent thread pool for block update)..."
	$(call run-sim,$(POOL_WY_TARGET))

pool-wy-gemm: $(POOL_WY_GEMM_TARGET)

$(POOL_WY_GEMM_TARGET): $(POOL_WY_GEMM_SRC) | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/qr_pool_hh_wy_gemm.o $(POOL_WY_GEMM_SRC)
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/qr_pool_hh_wy_gemm.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built pool-wy-gemm:  $@"

sim-pool-wy-gemm: pool-wy-gemm cfgs $(INPARAMS_FILE)
	@echo "Running pool-wy-gemm variant (GEMM-based WY-blocked trailing update)..."
	$(call run-sim,$(POOL_WY_GEMM_TARGET))

# Row-major binaries read ./../inparams1.txt; keep separate from the
# column-major inparams.txt so the two can be configured independently.
INPARAMS_RM_FILE = $(BUILD_DIR)/inparams1.txt

$(INPARAMS_RM_FILE): | $(HEXAGON_DIR)
	@if [ ! -f $(INPARAMS_RM_FILE) ]; then \
		echo "$(INPARAMS_DEFAULT)" > $(INPARAMS_RM_FILE); \
		echo "Created default $(INPARAMS_RM_FILE): $(INPARAMS_DEFAULT)"; \
	fi

sim-rowmajor: rowmajor cfgs $(INPARAMS_RM_FILE)
	@echo "Running unblocked row-major baseline on Hexagon simulator..."
	@echo "----- inparams1.txt -----"; cat $(INPARAMS_RM_FILE); echo "-------------------------"
	$(HEXAGON_SIM) -m$(HEX_VERSION) --simulated_returnval --timing \
		--usefs $(HEXAGON_DIR) \
		--pmu_statsfile $(HEXAGON_DIR)/pmu_stats.txt \
		--cosim_file $(Q6SS_CFG) \
		--l2tcm_base 0xd800 \
		--subsystem_base 0xFC90 \
		--rtos $(OSAM_CFG) $(QURT_OS) -- \
		$(ROWMAJOR_TARGET)

sim-rowmajor-wy: rowmajor-wy cfgs $(INPARAMS_RM_FILE)
	@echo "Running WY-blocked QR (row-major) on Hexagon simulator..."
	@echo "----- inparams1.txt -----"; cat $(INPARAMS_RM_FILE); echo "-------------------------"
	$(HEXAGON_SIM) -m$(HEX_VERSION) --simulated_returnval --timing \
		--usefs $(HEXAGON_DIR) \
		--pmu_statsfile $(HEXAGON_DIR)/pmu_stats.txt \
		--cosim_file $(Q6SS_CFG) \
		--l2tcm_base 0xd800 \
		--subsystem_base 0xFC90 \
		--rtos $(OSAM_CFG) $(QURT_OS) -- \
		$(ROWMAJOR_WY_TARGET)

# =============================================================================
# HVX Householder research-plan variants (v1..v6)
#   see hvx_householder_research_plan.md
# Each variant is a standalone source at repo root that reads
# build/inparams.txt (m n nt chunk use_static nb) and uses include/hvx_utils.h.
#   build a variant:  make v3_persistent_hvx
#   run a variant:    make sim-v3_persistent_hvx
#   build all:        make variants
# =============================================================================
VARIANTS = v1_hvx_reduction v2_hvx_pivot v3_persistent_hvx \
           v4_adaptive_chunk v5_dag_pipeline v6_combined

VARIANT_TARGETS = $(addprefix $(HEXAGON_DIR)/,$(VARIANTS))

.PHONY: variants $(VARIANTS) $(addprefix sim-,$(VARIANTS))

variants: $(VARIANT_TARGETS)

# Build rule: build/hexagon/vN  from  vN.c (+ shared header)
$(VARIANT_TARGETS): $(HEXAGON_DIR)/%: %.c include/hvx_utils.h | $(HEXAGON_DIR)
	@command -v $(CC_HEXAGON) > /dev/null 2>&1 || { \
		echo "ERROR: $(CC_HEXAGON) not found in PATH. Source the Hexagon SDK env."; exit 1; }
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/$*.o $<
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/$*.o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built variant:  $@"

# Convenience short name: make v3_persistent_hvx
$(VARIANTS): %: $(HEXAGON_DIR)/%

# Run rule: make sim-v3_persistent_hvx
$(addprefix sim-,$(VARIANTS)): sim-%: $(HEXAGON_DIR)/% cfgs $(INPARAMS_FILE)
	@echo "Running variant $* ..."
	$(call run-sim,$(HEXAGON_DIR)/$*)

# ---------------- Clean ----------------
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned $(BUILD_DIR)"

# ---------------- Help ----------------
help:
	@echo "QR factorization Makefile (Hexagon QuRT + HVX)"
	@echo "==============================================="
	@echo ""
	@echo "Build targets:"
	@echo "  make                 - Build all 4 binaries"
	@echo "  make baseline        - Build qr_colmajor_mt (unblocked, col-major)"
	@echo "  make blocked         - Build qr_wy_blocked_mt (WY, col-major; runs both)"
	@echo "  make rowmajor        - Build qr_rowmajor_mt_block_alloc (unblocked, row-major)"
	@echo "  make rowmajor-wy     - Build qr_wy_blocked_rm (WY, row-major; runs both)"
	@echo ""
	@echo "Run targets (Hexagon simulator):"
	@echo "  make sim             - sim-blocked (default)"
	@echo "  make sim-blocked     - Col-major: unblocked + WY-blocked timing"
	@echo "  make sim-baseline    - Col-major unblocked only"
	@echo "  make sim-rowmajor    - Row-major unblocked only (legacy)"
	@echo "  make sim-rowmajor-wy - Row-major: unblocked + WY-blocked timing"
	@echo ""
	@echo "Other:"
	@echo "  make inparams     - Show current $(INPARAMS_FILE)"
	@echo "  make cfgs         - Generate osam.cfg / q6ss.cfg"
	@echo "  make clean        - Remove $(BUILD_DIR)/"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "inparams.txt format (whitespace-separated):"
	@echo "  m n nt chunk_size use_static nb"
	@echo "  - m must be a multiple of 32 (HVX float lane count)"
	@echo "  - nt: number of threads (<=8)"
	@echo "  - use_static: 0=dynamic counter, 1=static range (baseline only)"
	@echo "  - nb: WY block width (1..32, blocked binary only)"
	@echo ""
	@echo "Default created on first run: $(INPARAMS_DEFAULT)"
	@echo ""
	@echo "Environment overrides:"
	@echo "  HEXAGON_SDK_ROOT  - Hexagon SDK path"
	@echo "  HEX_VERSION       - Hexagon arch (default: v69)"
