# Silent-Failure Diagnosis

Each failure type below gives the same output: `✓ Done` and an empty field. Do the
checks in this sequence. Each check is one typed command. Do not use `eval`.

## 1. Disabled — a Control Is the Gate

```bash
agent-browser is enabled "#field"
```

If the result is false, find the gate control. Set the gate. Then fill the field
again.

Some fields are conditional. A gate checkbox or a gate select enables them. These
fields frequently have no accessible name in the snapshot. Use `check "#gate"`. The
`disabled` attribute changes from true to false. Then `fill` operates correctly. Read
the question text near the field to find the gate.

## 2. Hidden — an Upstream Question Has No Answer, or the Field Is Correctly Hidden

```bash
agent-browser is visible "#field"
```

If the result is false, a parent element has `display:none`. The input itself shows
`disabled=false`. This failure looks the same as the mask failure. It is not the same.
There are two possible causes:

- An upstream master gate has no answer. Answer the gate first. Fill from the top of
  the form to the bottom.
- The block is correctly hidden. Example: "Mailing address the same as residential?"
  with the answer "Yes" hides the mailing block. An empty hidden block is correct. Do
  not fill it. Read the text of the gate question before you decide.

## 3. Masked — the Value Shows `__/__/____` or `(___) ___-____`

The mask JavaScript monitors real keydown and keypress events. The `fill` and `type`
commands make synthetic events. The mask ignores these events or changes their
sequence. Then the mask removes the value. The command shows success.

Do the steps below in sequence. Use `get value` after each step.

1. Use `fill "#field" "01152000"`. Use only characters, with no separators. The mask
   adds the separators.
2. If the field is empty, use `fill "#field" ""` and then `type "#field" "…"`.
   CAUTION: some masks move the caret after each keystroke. On these masks, `type`
   reverses the input. Example: a 5-digit ZIP `ABCDE` becomes `EDCBA`.
3. If the field is empty or the value is reversed, use one `key` command for each
   character. This is the only method that all keydown masks accept:

```bash
agent-browser click "#field"          # Set the focus first
agent-browser key "Meta+a"            # Use "Control+a" when not on macOS
agent-browser key "Backspace"        
agent-browser key "Home"             
agent-browser key "0"                 # One key command for each character
agent-browser key "1"
agent-browser get value "#field"      # Expect the value in the mask format
```

The select-all, delete, and Home steps clear the field with real keystrokes. The
`fill ""` command can put the internal buffer of a mask in a bad state. Then the field
ignores keys, or the caret stops at the end and only the last position gets a
character. The keyboard clear steps correct this state. The steps cause no damage on a
clean field. Always include them.

**A mask reformats the value. Compare the digits, not the strings.** You send digits
only. The readback gives the mask format. Examples confirmed in one run: nine digits
came back as `NNN-NN-NNNN`, `MMDDYYYY` came back as `MM/DD/YYYY`, and ten digits came
back as `(NNN) NNN-NNNN`. A string comparison of the sent value and the readback shows
a false failure. Remove the non-digit characters from both, then compare.

Not every text field has a mask. A field with `type=text` and a large `maxlength` (for
example 255) frequently accepts a plain `fill` with the separators included. Try
`fill` first and read the value back. Use the per-key method only after the readback
fails.

Mask sizes give the format. A date field with `maxLength=8` accepts `MMDDYYYY`. A
telephone field with `maxLength=10` accepts ten digits. An SSN field with `maxLength=9`
accepts nine digits. A state field with `maxLength=2` accepts the two-letter
abbreviation.

## 4. maxLength — the Field Refused or Cut the Value

```bash
agent-browser get attr "#field" maxlength
```

A value longer than `maxLength` can be refused as one unit, not cut. Example:
"California" in a two-character field gives an empty field. Make the value shorter.
Use "CA" for the state. Add the unit or apartment to the street line, within its
limit.

## General Rules

- **XPath is not supported.** A command with an XPath selector (`//input[@id=…]`) can
  show `✓ Done` and do nothing. This is a silent no-op, in the same class as the mask
  failure. Use CSS selectors only.
- **A selector with `[id=X]` finds only the first element with that id.** Some sites
  give one id to a full checkbox group (invalid HTML). Then `get count` shows 1,
  probes for the other members show "not found", and a compound selector such as
  `input[id='X'][value='96']` fails for all members after the first. Use the `FIELDS`
  helper in `scripts/fill-helpers.sh` to get the full group map in one call. Set a
  member with `input[value='N']` alone, or by its visible label:
  `find role checkbox check --name "<label>"`. Put the flags AFTER the action —
  `find role checkbox --name "…" check` fails with an "unknown action" error.

- The `click` command changes a checkbox state each time. Use `check` and `uncheck`.
  These commands are idempotent. After you write a Yes/No pair, read both checkboxes
  again. The two are independent. Both can become unchecked.
- An old snapshot becomes incorrect after the DOM changes. Do not read the old
  snapshot output again.
- **`get text` can return a stale string for one call after a click that changes the
  DOM.** Confirmed on an expander: immediately after `click "div.header"`,
  `get text` still gave "+ Expand" and `is enabled "#submit"` still gave false, while
  `get html "div.header"` already gave "- Collapse". A second read of both agreed. Use
  `get html` as the authoritative read after a click. Corroborate with
  `is visible` on the content that the click shows. Do not report a bug from one
  `get text` result after a click: read it two times.
- The `get value` command on a `<select>` returns the option index, not the text.
- **`get attr` prints `✓ Done` and no value when the attribute is absent.** The
  command found the element. The attribute does not exist. Do not read `✓ Done` as
  the value of the attribute. Example from a probe of nine fields: four fields gave
  `required=required`, and five gave `required=✓ Done`, which means "not required".
  A `maxlength` probe on a `<select>` gives `✓ Done` for the same reason. To get a
  count of the elements that have the attribute, use `get count "#field[required]"`
  instead.
- **`:nth-of-type(N)` fails for N greater than 1 when the matched elements have
  different parents.** The `:nth-of-type` pseudo-class counts inside one parent, not
  across the document. On a page where `get count "iframe"` gave 6 and
  `get count "form"` gave 4, only `iframe:nth-of-type(1)` and `form:nth-of-type(1)`
  returned a result. Elements 2 to 6 gave "Element not found". Do not use
  `:nth-of-type` to make a list of the elements on a page. Use `get html` on the
  container and parse the output (see the `FIELDS` helper in
  `scripts/fill-helpers.sh`), or use `find role`.
- To get all the options of a `<select>`, use `get html` on the select and extract the
  `<option>` tags. Do not use `get text`: the options come out as one text block that
  is difficult to separate.
- The `connect <port>` command prints `[agent-browser] launched browser` when it
  attaches to a Chrome that already runs. It prints `[agent-browser] relaunched
  browser` when the session had a browser before. Both messages can occur on a
  correct attachment. Use `get cdp-url` to make sure that the attachment is correct:
  the URL must contain the port that you gave.

## The Screenshot Is Not Evidence

A screenshot can show a false failure. Use `get value`, `is checked`, and `is visible`
to verify a write. Use the screenshot only to find a control that you cannot find in
the field map.

- **A sticky site header covers the fields under it.** In `screenshot --full`, the
  header stays at the top of the image and hides one or more rows of the form. A
  correctly filled field can look empty. This was confirmed on one form: the Date of
  Birth row was behind the header, and `get value` gave the correct value.
- **A reCAPTCHA widget frequently does not appear in the screenshot.** The widget is in
  an iframe. The absence of the widget in the image is not proof that the page has no
  captcha. Use `get count "iframe"` and read the iframe `src`, or
  `is visible "#g-recaptcha-response"`.
- **The state of the submit button proves nothing about validation.** A server-side
  validated form (example: a Drupal webform with a captcha) keeps the submit button
  enabled with an empty form and no captcha token. `is enabled "#submit"` returns true
  in both a valid and an invalid state.
- **The token field is the only reliable bot-check signal. The iframe count is not.**
  Read the value of the hidden token input:
  - reCAPTCHA: `get value "#g-recaptcha-response"`
  - Cloudflare Turnstile: `get value "[name='cf-turnstile-response']"`

  An empty value means no token: a human must complete the challenge before the
  submit. A long value (several hundred characters) means the challenge passed.
  `get count "iframe"` gave 0 on one page at the same time that a valid ~750-character
  Turnstile token was present. Do not use the iframe count to decide if a challenge
  ran or failed.

## Fields You Must Not Fill

The field inventory shows more fields than the user must answer. Remove these before
you make the gap analysis. A field in this group that has a value is an error, not
progress.

- **A honeypot.** A field with the label "Leave this field blank", or a hidden field
  with a name such as `url`, is an anti-spam trap. A value in this field marks the
  submission as a robot. Leave it empty. Example seen: `#edit-url` on a Drupal
  webform.
- **The site search box.** The header and the mobile menu each have a search input.
  These inputs are in the `FIELDS "body"` output, but they are not part of the
  application. Example ids: `header-search-input`, `mobile-search-keys`.
- **A field in an advertisement iframe or a reCAPTCHA iframe.** Read the `src`
  attribute of each iframe before you go inside one. A `google.com/recaptcha` src or
  an ad-tracker src has no application field. On one page, 6 iframes were all
  reCAPTCHA and ad trackers, and the application form was inline in the page.
- **A "if different from…" field when the two values are the same.** Example: a
  mailing address field with the label "if different from home address". An empty
  field is the correct answer. Do not repeat the home address.
- **reCAPTCHA is a submit blocker, not a field.** Report it in the gap analysis as a
  blocker. Do not attempt to solve it. The user must complete it, or the user must
  submit the form.
