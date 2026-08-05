# Field Type Patterns

Command examples for the most common form controls. Load when you need an exact action shape.

> **`fill` first, always — including masked fields.** `fill` clears the field and sets the value in one step. `type` does NOT clear (it appends), and on masks that reposition the caret per keystroke it reverses the input: typing `92595` into a zip mask produces `59529`.
>
> Use `type` only when `fill` leaves a field empty or unformatted, and clear it first:
>
> ```json
> ["fill", "@e1", ""]
> ["type", "@e1", "92595"]
> ["get", "value", "@e1"]
> ```

## Text Fields (use `fill`)

```json
["fill", "@e3", "John Doe"]
["fill", "#firstNameTxt", "John Doe"]
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
["select", "#genderIdentityDrpDwn", ""]
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
["snapshot", "-s", "main"]
```
