# ------------------------------------------------------------
# docs-rag Makefile (dev build + JS test + clean packaging)
# ------------------------------------------------------------

PLUGIN_NAME = docs-rag
VERSION = 1.1.0

DIST_DIR = dist
STAGE_DIR = stage
PKG_FILE = $(PLUGIN_NAME)-$(VERSION).tgz

TEST_DIR = test

.PHONY: all build test clean stage stage-install stage-build stage-pack prune package

# ------------------------------------------------------------
# Default: build + test for development
# ------------------------------------------------------------
all: build test

# ------------------------------------------------------------
# Build inside dev folder (no deletion of src)
# ------------------------------------------------------------
build:
	npm ci
	npm run build

# ------------------------------------------------------------
# JS test: validate dist/index.js loads and exports correctly
# ------------------------------------------------------------
test: build
	@echo "Running JS plugin validation..."
	@node -e "import('./dist/index.js').then(m=>{ \
	    if(!m.default) { console.error('✘ no default export'); process.exit(1); } \
	    if(typeof m.default.setup !== 'function') { console.error('✘ setup is not a function'); process.exit(1); } \
	    console.log('✔ plugin object loaded'); \
	    console.log('✔ setup() is a function'); \
	}).catch(e=>{console.error(e); process.exit(1);});"
	@echo "✔ JS validation passed."
	@node $(TEST_DIR)/sanity.js || { echo "✘ Smoke test failed."; exit 1; }

# ------------------------------------------------------------
# Clean dist + stage
# ------------------------------------------------------------
clean:
	rm -rf $(DIST_DIR)
	rm -rf $(STAGE_DIR)
	mkdir -p $(DIST_DIR)

# ------------------------------------------------------------
# Create staging copy for packaging
# ------------------------------------------------------------
stage:
	mkdir -p $(STAGE_DIR)
	rsync -av \
	--include="index.js" \
        --include="AGENTS.md" \
        --include="README.md" \
        --include="config.json" \
        --include="package.json" \
        --include="package-lock.json" \
        --include="tsconfig.json" \
        --include="src/***" \
        --include="python/***" \
        --exclude="*" \
        ./ $(STAGE_DIR)/
	# Remove Python caches
	find $(STAGE_DIR)/python -type d -name "__pycache__" -exec rm -rf {} +
	find $(STAGE_DIR)/python -type f -name "*.pyc" -delete
	find $(STAGE_DIR)/python/rag -type d -name "__pycache__" -exec rm -rf {} +
	find $(STAGE_DIR)/python/rag -type f -name "*.pyc" -delete

# ------------------------------------------------------------
# Install full deps inside staging
# ------------------------------------------------------------
stage-install:
	cd $(STAGE_DIR) && npm ci

# ------------------------------------------------------------
# Build TypeScript inside staging
# ------------------------------------------------------------
stage-build:
	cd $(STAGE_DIR) && npm run build
	rm -rf $(STAGE_DIR)/src

# ------------------------------------------------------------
# Package from staging → dist
# ------------------------------------------------------------
stage-pack:
	cd $(STAGE_DIR) && npm pack
	mv $(STAGE_DIR)/$(PKG_FILE) $(DIST_DIR)/

# ------------------------------------------------------------
# Prune: delete staging directory
# ------------------------------------------------------------
prune:
	rm -rf $(STAGE_DIR)

# ------------------------------------------------------------
# Full packaging pipeline
# ------------------------------------------------------------
package: clean stage stage-install stage-build stage-pack prune

