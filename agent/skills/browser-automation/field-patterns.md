# Field Type Patterns

Argv examples for the most common form controls. Load when you need an exact command shape.

## Text Fields (use `fill`)

```json
["fill", "@e3", "John Doe"]
["fill", "#firstNameTxt", "John Doe"]
```

## Date Fields (use `type`)

Check `maxlength`. If `maxlength="8"`, use digits only (MMDDYYYY). Click, select any existing content, then type:

```json
["click", "@e1"]
["press", "Control+a"]
["type", "@e1", "01152000"]
["get", "value", "@e1"]
```

Or if using a date picker:

```json
["click", "@e1"]
["snapshot", "-i"]
["click", "@e5"]
```

## SSN Fields (use `type`)

Check `maxlength`. If `maxlength="9"`, digits only:

```json
["click", "@e1"]
["press", "Control+a"]
["type", "@e1", "123456789"]
["get", "value", "@e1"]
```

## Phone Number Fields (use `type`)

Check `maxlength`. If `maxlength="10"`, digits only:

```json
["click", "@e1"]
["press", "Control+a"]
["type", "@e1", "5551234567"]
["get", "value", "@e1"]
```

## State Fields (use `type`)

Check `maxlength`. If `maxlength="2"`, use abbreviation:

```json
["click", "@e1"]
["press", "Control+a"]
["type", "@e1", "CA"]
["get", "value", "@e1"]
```

## Native Dropdowns (select)

```json
["select", "@e1", "Option Value"]
["select", "#genderIdentityDrpDwn", "57"]
```

## Checkboxes

```json
["check", "@e1"]
["uncheck", "@e1"]
["check", "#chkBxApplyYourselfYes"]
```

## Radio Buttons

```json
["click", "@e1"]
```

ALWAYS re-snapshot after a radio click — radio selections often reveal conditional fields:

```json
["snapshot", "-s", "form"]
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
