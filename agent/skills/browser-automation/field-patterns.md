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
