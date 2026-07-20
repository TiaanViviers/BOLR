include config.mk.example
-include config.mk

BUILD ?= debug
BUILD_DIR ?= build
ROOT := $(CURDIR)
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
TOOLS_DIR := tools
BENCH_DIR := benchmarks
OUT_DIR := $(BUILD_DIR)/$(BUILD)
LIB_STATIC := $(OUT_DIR)/libbolr.a
LIB_SHARED := $(OUT_DIR)/libbolr.so
TEST_BIN := $(OUT_DIR)/tests/bolr_tests
GOLDEN_BIN := $(OUT_DIR)/tools/bolr_golden_runner
ABI_BIN := $(OUT_DIR)/tools/bolr_abi_info
BENCH_BIN := $(OUT_DIR)/benchmarks/bolr_bench

WARN_FLAGS := -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wformat=2 -Wundef
BASE_CFLAGS := -std=c11 $(WARN_FLAGS) -fPIC
DEBUG_CFLAGS := -O0 -g3 -fno-omit-frame-pointer
RELEASE_CFLAGS := -O3 -DNDEBUG
SANITIZE_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer

ifeq ($(BUILD),debug)
  BUILD_CFLAGS := $(DEBUG_CFLAGS)
else ifeq ($(BUILD),release)
  BUILD_CFLAGS := $(RELEASE_CFLAGS)
else ifeq ($(BUILD),sanitize)
  BUILD_CFLAGS := $(DEBUG_CFLAGS) $(SANITIZE_FLAGS)
  LDFLAGS += $(SANITIZE_FLAGS)
endif

ALL_CFLAGS := $(BASE_CFLAGS) $(BUILD_CFLAGS) $(CPPFLAGS) $(CFLAGS)
ALL_LDLIBS := $(LDLIBS) $(BLAS_LIBS) $(LAPACK_LIBS) -lm

SRC_FILES := \
  $(SRC_DIR)/version.c \
  $(SRC_DIR)/status.c \
  $(SRC_DIR)/allocator.c \
  $(SRC_DIR)/array.c \
  $(SRC_DIR)/math.c \
  $(SRC_DIR)/linalg.c \
  $(SRC_DIR)/state_layout.c \
  $(SRC_DIR)/score_dense.c \
  $(SRC_DIR)/score_context.c \
  $(SRC_DIR)/score_graph.c \
  $(SRC_DIR)/score_composite.c \
  $(SRC_DIR)/observation_candidate_a.c \
  $(SRC_DIR)/observation_candidate_b_exact.c \
  $(SRC_DIR)/ordered_partition.c \
  $(SRC_DIR)/penalty.c \
  $(SRC_DIR)/target_candidate_a.c \
  $(SRC_DIR)/standardizer.c \
  $(SRC_DIR)/surprise.c \
  $(SRC_DIR)/bocpd.c \
  $(SRC_DIR)/attribution.c \
  $(SRC_DIR)/reset.c \
  $(SRC_DIR)/adaptation.c \
  $(SRC_DIR)/prediction.c \
  $(SRC_DIR)/region.c \
  $(SRC_DIR)/decision.c \
  $(SRC_DIR)/dynamics.c \
  $(SRC_DIR)/workspace.c \
  $(SRC_DIR)/checkpoint_schema.c \
  $(SRC_DIR)/rng.c \
  $(SRC_DIR)/gaussian_state.c \
  $(SRC_DIR)/gaussian_prediction.c \
  $(SRC_DIR)/score_sampling.c \
  $(SRC_DIR)/replay.c \
  $(SRC_DIR)/endian.c \
  $(SRC_DIR)/crc32.c \
  $(SRC_DIR)/checkpoint_sections.c \
  $(SRC_DIR)/checkpoint_codec.c \
  $(SRC_DIR)/checkpoint_file.c \
  $(SRC_DIR)/posterior_objective.c \
  $(SRC_DIR)/newton.c \
  $(SRC_DIR)/line_search.c \
  $(SRC_DIR)/laplace.c \
  $(SRC_DIR)/static_fit.c
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/obj/%.o,$(SRC_FILES))

TEST_SRC := \
  $(TEST_DIR)/test_main.c \
  $(TEST_DIR)/test_math.c \
  $(TEST_DIR)/test_linalg.c \
  $(TEST_DIR)/test_ownership.c \
  $(TEST_DIR)/test_state_layout.c \
  $(TEST_DIR)/test_score.c \
  $(TEST_DIR)/test_candidate_a.c \
  $(TEST_DIR)/test_candidate_a_target.c \
  $(TEST_DIR)/test_ordered_partition.c \
  $(TEST_DIR)/test_candidate_b_exact.c \
  $(TEST_DIR)/test_candidate_b_sampled.c \
  $(TEST_DIR)/test_candidate_b_laplace.c \
  $(TEST_DIR)/test_standardizer.c \
  $(TEST_DIR)/test_bocpd.c \
  $(TEST_DIR)/test_attribution.c \
  $(TEST_DIR)/test_partial_reset.c \
  $(TEST_DIR)/test_adaptive_policy.c \
  $(TEST_DIR)/test_adaptive_checkpoint.c \
  $(TEST_DIR)/test_score_uncertainty.c \
  $(TEST_DIR)/test_pairwise_probability.c \
  $(TEST_DIR)/test_decision_policy.c \
  $(TEST_DIR)/test_consensus_region.c \
  $(TEST_DIR)/test_connected_components.c \
  $(TEST_DIR)/test_region_summary.c \
  $(TEST_DIR)/test_graph_medoid.c \
  $(TEST_DIR)/test_decision_calibration.c \
  $(TEST_DIR)/test_penalty.c \
  $(TEST_DIR)/test_dynamics.c \
  $(TEST_DIR)/test_golden.c \
  $(TEST_DIR)/test_gaussian_state.c \
  $(TEST_DIR)/test_gaussian_prediction.c \
  $(TEST_DIR)/test_rng.c \
  $(TEST_DIR)/test_sampling.c \
  $(TEST_DIR)/test_rank_accumulator.c \
  $(TEST_DIR)/test_replay.c \
  $(TEST_DIR)/test_checkpoint_header.c \
  $(TEST_DIR)/test_checkpoint_sections.c \
  $(TEST_DIR)/test_checkpoint_codec.c \
  $(TEST_DIR)/test_checkpoint_corruption.c \
  $(TEST_DIR)/test_checkpoint_file.c \
  $(TEST_DIR)/test_checkpoint_restart.c \
  $(TEST_DIR)/test_posterior_objective.c \
  $(TEST_DIR)/test_newton.c \
  $(TEST_DIR)/test_laplace.c \
  $(TEST_DIR)/test_static_fit.c \
  $(TEST_DIR)/test_inference_ownership.c
TEST_OBJ := $(patsubst $(TEST_DIR)/%.c,$(OUT_DIR)/obj/tests/%.o,$(TEST_SRC))

TOOL_SRC := $(TOOLS_DIR)/golden_runner.c $(TOOLS_DIR)/abi_info.c
TOOL_OBJ := $(patsubst $(TOOLS_DIR)/%.c,$(OUT_DIR)/obj/tools/%.o,$(TOOL_SRC))

BENCH_SRC := $(BENCH_DIR)/bench_score.c $(BENCH_DIR)/bench_candidate_a.c
BENCH_OBJ := $(patsubst $(BENCH_DIR)/%.c,$(OUT_DIR)/obj/benchmarks/%.o,$(BENCH_SRC))

.PHONY: all static shared tests test golden bench debug release sanitize install uninstall clean distclean print-config

all: static shared
static: $(LIB_STATIC)
shared: $(LIB_SHARED)
tests: $(TEST_BIN)
test: tests
	$(TEST_BIN)
golden: $(GOLDEN_BIN)
bench: $(BENCH_BIN)
debug:
	$(MAKE) BUILD=debug all
release:
	$(MAKE) BUILD=release all
sanitize:
	$(MAKE) BUILD=sanitize clean
	$(MAKE) BUILD=sanitize all tests
	ASAN_OPTIONS=detect_leaks=0 $(BUILD_DIR)/sanitize/tests/bolr_tests
install: all
	install -d $(DESTDIR)$(PREFIX)/include/bolr $(DESTDIR)$(PREFIX)/lib
	cp -r $(INCLUDE_DIR)/bolr/*.h $(DESTDIR)$(PREFIX)/include/bolr/
	install -m 644 $(LIB_STATIC) $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(DESTDIR)$(PREFIX)/lib/
uninstall:
	rm -f $(DESTDIR)$(PREFIX)/lib/libbolr.a $(DESTDIR)$(PREFIX)/lib/libbolr.so
	rm -rf $(DESTDIR)$(PREFIX)/include/bolr
clean:
	rm -rf $(BUILD_DIR)
distclean:
	rm -rf build
	rm -f config.mk
print-config:
	@echo BUILD=$(BUILD)
	@echo CC=$(CC)
	@echo PREFIX=$(PREFIX)
	@echo BUILD_DIR=$(BUILD_DIR)
	@echo OUT_DIR=$(OUT_DIR)
	@echo LIB_SHARED=$(LIB_SHARED)
	@echo ALL_CFLAGS=$(ALL_CFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo ALL_LDLIBS=$(ALL_LDLIBS)
	@echo BLAS_LIBS=$(BLAS_LIBS)
	@echo LAPACK_LIBS=$(LAPACK_LIBS)

$(OUT_DIR)/obj/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(ALL_CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OUT_DIR)/obj/tests/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(ALL_CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OUT_DIR)/obj/tools/%.o: $(TOOLS_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(ALL_CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OUT_DIR)/obj/benchmarks/%.o: $(BENCH_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(ALL_CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(LIB_STATIC): $(OBJ_FILES)
	@mkdir -p $(@D)
	$(AR) rcs $@ $(OBJ_FILES)
	$(RANLIB) $@

$(LIB_SHARED): $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CC) -shared $(LDFLAGS) -o $@ $(OBJ_FILES) $(ALL_LDLIBS)

$(TEST_BIN): $(OBJ_FILES) $(TEST_OBJ)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES) $(TEST_OBJ) $(ALL_LDLIBS)

$(GOLDEN_BIN): $(OBJ_FILES) $(OUT_DIR)/obj/tools/golden_runner.o
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES) $(OUT_DIR)/obj/tools/golden_runner.o $(ALL_LDLIBS)

$(ABI_BIN): $(OBJ_FILES) $(OUT_DIR)/obj/tools/abi_info.o
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES) $(OUT_DIR)/obj/tools/abi_info.o $(ALL_LDLIBS)

$(BENCH_BIN): $(OBJ_FILES) $(BENCH_OBJ)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES) $(BENCH_OBJ) $(ALL_LDLIBS)
