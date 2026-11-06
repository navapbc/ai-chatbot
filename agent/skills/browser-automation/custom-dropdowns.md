# Custom Dropdowns

Load this reference when a native `select` command fails or has no effect, or when the snapshot shows `select2-container` or `chosen-container` classes.

The `select` command ONLY works on native `<select>` elements. Custom dropdown widgets (Select2, Chosen) render styled HTML instead.

**How to detect:** `select` fails or has no effect; snapshot shows `<span>` or `<div>` with classes like `select2-container`, `chosen-container`.

## Pattern for Select2/Chosen

```json
// 1. Click the dropdown trigger
["click", "@e5"]
// 2. Wait for the panel to render
["wait", "300"]
// 3. Snapshot to see options
["snapshot", "-i"]
// 4. Click the desired option
["click", "@e12"]
// 5. Re-snapshot (DOM changed)
["snapshot", "-s", "form"]
```

## Select2 with Search (common in Drupal)

```json
// 1. Click to open
["click", "@e5"]
["wait", "300"]
// 2. Type into search (auto-focused in Select2)
["type", ":focus", "Riverside"]
["wait", "300"]
// 3. Snapshot filtered results
["snapshot", "-i"]
// 4. Click match
["click", "@e12"]
// 5. Re-snapshot
["snapshot", "-s", "form"]
```

## Drupal Tips

- Always use `["snapshot", "-s", "form"]` after the initial full snapshot — Drupal pages have heavy nav/sidebar/footer.
- Drupal webforms frequently use Select2 for dropdowns with many options (clinics, locations, languages).
- If clicking the trigger opens a search input inside the dropdown, type into `:focus` rather than finding the search input's ref.
