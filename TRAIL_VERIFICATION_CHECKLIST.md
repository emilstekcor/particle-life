# GPU Trail Verification Checklist

## Debug Infrastructure Added
- Console output every ~1 second showing: particles, trail_len, head, valid_len, enabled, vertex_count
- Debug mode cap (32 particles, trail length 8) - set `debug_cap = true` in draw.rs line 566
- UI trace_len slider now wired to actual GPU runtime

## Verification Tests

### 1. Visual Sanity Test
**Setup:**
- Spawn 10 particles
- Set velocity to slow (0.1 or lower)
- Trail length 8-16
- Use step-once mode for precise control

**Expected Results:**
- Each step adds exactly one segment per particle
- Oldest segment drops off when trail_len exceeded
- No teleport lines to origin
- No lines connecting different particles
- Debug output shows vertex_count = particles * (trail_len-1) * 2

### 2. Ring Order Test
**Setup:**
- Single particle only
- Move in simple direction (e.g., along X axis)
- Use step-once mode
- Trail length 8

**Expected Results:**
- Trail appears as ordered chain behind particle
- No scrambled or reversed segments
- Each new segment connects to previous
- Trail head increments correctly (debug output shows head cycling 0-7)

### 3. Pause Behavior Test
**Setup:**
- Enable trails
- Pause simulation
- Move camera around (UI frames continue)

**Expected Results:**
- No new trail points added while paused
- Debug output shows no head increment during pause
- UI redraws don't extend trails
- Trail remains static during pause

### 4. Reset Behavior Test
**Setup:**
- Active trails with history
- Click "Clear" button

**Expected Results:**
- All trail segments disappear immediately
- No stale segments remain
- Debug output shows valid_len resets to 0
- First few steps after respawn "warm up" correctly via valid_len

### 5. Trail Length Control Test
**Setup:**
- Active trails
- Adjust trace_len slider from 4 to 32

**Expected Results:**
- Trail segments immediately adjust to new length
- Debug output shows trail_len changes
- Vertex count updates accordingly
- No lag or mismatch between UI and GPU

### 6. High Count Performance Test
**Setup:**
- Test with 1k, 10k, 50k particles
- Trail length 16
- Monitor frame time and debug output

**Expected Results:**
- Frame time scales predictably
- No visual glitches or artifacts
- Debug output shows correct vertex counts
- Alpha blending doesn't become overwhelming

## Critical Failure Points to Watch For

### Ring Math Issues
- `slot_for_logical_index()` off by one
- Trail head advanced at wrong time
- Capture before/after head increment mismatch

### Bind Group Mismatches
- Trail shader bindings don't match bind group layout
- Wrong buffer types (uniform vs storage)
- Binding indices don't match

### Draw Count Mismatches
- CPU vertex count doesn't match shader expectations
- valid_segments calculation wrong
- Particle index instability causing jumps

### State Synchronization
- UI trace_len not synced to GPU
- Trail enable flag not working
- Stale trail history after operations

## Debug Mode Usage
Set `debug_cap = true` in `src/renderer/draw.rs` line 566 to:
- Limit to first 32 particles
- Force trail length 8
- Make visual inspection much easier

## Success Criteria
All tests pass with:
- Correct visual behavior
- Expected debug output values
- No performance anomalies
- No visual glitches or artifacts
