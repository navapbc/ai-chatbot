# Field Type Patterns

Command examples for the most common form controls. Load when you need an exact action shape.

> **`fill` first, then ALWAYS verify with `get value`.** Masks can silently reject `fill` — the command reports success while the field reverts to `__/__/____`. The readback is the only way to know.
>
> **Escalation ladder for masked fields** (a value showing `_` placeholders after fill means the mask listens for real keystrokes):
>
> 1. `["fill", "@e1", "01152000"]` then `["get", "value", "@e1"]` — if the value stuck, done.
> 2. If empty/placeholder: clear and `type` — `["fill", "@e1", ""]`, `["type", "@e1", "92595"]`, verify again. Caution: on masks that reposition the caret per keystroke, `type` reverses input (`92595` → `59529`).
> 3. If still rejected or reversed: **focus, then one `key` per character** — this is the only method keydown-listening masks accept:
>
> ```json
> ["click", "@e1"]
> ["key", "0"]
> ["key", "1"]
> ["key", "1"]
> ["key", "5"]
> ["key", "2"]
> ["key", "0"]
> ["key", "0"]
> ["key", "0"]
> ["get", "value", "@e1"]
> ```
>
> Send raw digits only — the mask inserts separators itself. Never proceed past a masked field without a verified readback.

## Text Fields (use `fill`)

```json
["fill", "@e3", "John Doe"]
["fill", "#firstName", "John Doe"]
```

## Date Fields

Check `maxlength`. If `maxlength="8"`, use digits only (MMDDYYYY). Fill it, then verify:

```json
["fill", "@e1", "01152000"]
["get", "value", "@e1"]
```

Or if using a date picker:

```json
["click", "@e1"]
["snapshot", "-i"]
["click", "@e5"]
```

## SSN Fields

Check `maxlength`. If `maxlength="9"`, digits only:

```json
["fill", "@e1", "123456789"]
["get", "value", "@e1"]
```

## Phone Number Fields

Check `maxlength`. If `maxlength="10"`, digits only:

```json
["fill", "@e1", "5551234567"]
["get", "value", "@e1"]
```

## State Fields

Check `maxlength`. If `maxlength="2"`, use abbreviation:

```json
["fill", "@e1", "CA"]
["get", "value", "@e1"]
```

## Native Dropdowns (select)

```json
["select", "@e1", ""]
["select", "#languageSelect", ""]
```

## Checkboxes

```json
["check", "@e1"]
["uncheck", "@e1"]
["check", "#agreeYes"]
```

## Radio Buttons

```json
["click", "@e1"]
```

ALWAYS re-snapshot after a radio click — radio selections often reveal conditional fields:

```json
["snapshot", "-s", "main"]
```

## When a Masked Field Will Not Take

`type` writes nothing, or only the first character, when the mask reformats
asynchronously and drops keys arriving mid-reformat. Re-typing fails the same
way. Escalate once with a batched per-character sequence — ONE tool call:

```json
["batch", "--bail", "click @e1", "press Control+a", "press Delete",
 "press 1", "wait 150", "press 2", "wait 150", "press 3", "wait 150",
 "press 4", "wait 150", "press 5", "wait 150", "press 6", "wait 150",
 "press 7", "wait 150", "press 8", "wait 150", "press 9",
 "get value @e1"]
```

Read the last array element for the resulting value. If keys still drop, retry
once with `wait 300`, then stop: record what `get value` returned and report the
field for caseworker review rather than looping.
