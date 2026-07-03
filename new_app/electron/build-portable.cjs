const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const outputRoot = path.join(projectRoot, "dist-desktop");
const unpacked = path.join(outputRoot, "win-unpacked");
const unpackedTmp = path.join(outputRoot, "win-unpacked.tmp");

function assertInsideProject(target) {
  const resolved = path.resolve(target);
  if (!resolved.startsWith(projectRoot + path.sep)) {
    throw new Error(`Refusing to touch path outside project: ${resolved}`);
  }
  return resolved;
}

function removeGenerated(target) {
  const resolved = assertInsideProject(target);
  fs.rmSync(resolved, { recursive: true, force: true });
}

function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function renameWithRetry(source, destination) {
  let lastError = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      fs.renameSync(source, destination);
      return;
    } catch (error) {
      lastError = error;
      sleep(500);
    }
  }
  console.warn(`Rename did not settle quickly (${lastError && lastError.message}); copying unpacked app instead.`);
  removeGenerated(destination);
  fs.cpSync(source, destination, { recursive: true });
}

function electronBuilder(args) {
  const cli = require.resolve("electron-builder/cli.js");
  const result = spawnSync(process.execPath, [cli, ...args], {
    cwd: projectRoot,
    stdio: "inherit",
    shell: false
  });
  if (result.error) {
    console.error(result.error.message);
  }
  return result;
}

removeGenerated(unpacked);
removeGenerated(unpackedTmp);

const first = electronBuilder(["--win", "portable"]);
if (first.status === 0) {
  process.exit(0);
}

if (!fs.existsSync(unpackedTmp) || fs.existsSync(unpacked)) {
  process.exit(first.status || 1);
}

console.warn("electron-builder left win-unpacked.tmp after a failed rename; retrying with --prepackaged.");
renameWithRetry(assertInsideProject(unpackedTmp), assertInsideProject(unpacked));

const second = electronBuilder(["--win", "portable", "--prepackaged", unpacked]);
process.exit(second.status || 0);
