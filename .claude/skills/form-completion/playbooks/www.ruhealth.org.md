# Playbook: www.ruhealth.org (Riverside University Health System, New to WIC)

Confirmed 2026-08-05 with agent-browser 0.33.2.

**WARNING: This is a live government benefits form. Do not submit test data. Do not
submit without explicit approval from the user.**

**Freshness probe.** Each command below must return 1. If a command does not return 1,
the site changed. Use the cold-start procedure. Write this file again.

```bash
agent-browser get count "#edit-name"
agent-browser get count "#edit-please-choose-the-wic-clinic-closest-to-you"
agent-browser get count "#edit-submit"
```

## URL — the Form Is on the Entry Page

```
https://www.ruhealth.org/appointments/apply-4-wic-form#
```

The form is on the entry page. There are no pages between the entry URL and the form.
No apply link and no continue link is necessary.

The form is a Drupal webform. It is INLINE in the page. It is not in an iframe. The
page has 6 iframes: one reCAPTCHA anchor, one reCAPTCHA bframe, and ad trackers. Do
not look for the fields in an iframe.

The form element is `form.webform-submission-form`. Give this selector to the `FIELDS`
helper to get the field inventory in one call.

## Selector Rules for This Site

Use `#id` selectors. Every field has a stable id with the `edit-` prefix.

`iframe:nth-of-type(N)` and `form:nth-of-type(N)` FAIL for N greater than 1 on this
page. The elements do not have the same parent, so `nth-of-type` does not count them
as one group. Use `#id` selectors, or attribute selectors such as
`iframe[src*='recaptcha/api2/anchor']`.

## Required Fields

Three fields have the HTML `required` attribute:

- `#edit-date-of-birth`
- `#edit-home-address`
- `#edit-mobile`

`#edit-name` and `#edit-email` are NOT HTML-required. To test this, use
`get attr "#edit-name" required`. The command returns `✓ Done` when the attribute is
absent, and `required` when the attribute is present. `✓ Done` is not a value.

The usual payload contains a value for each required field. There is no gap question
for the required fields on this site. Ask the participant instead for the decisions in
the "Gates" section, because the payload does not contain those.

## Field Table

All the text fields have `maxlength 255`, except where the table gives another value.
No field on this form has an input mask. `fill` works on every text field, including
the date field.

| id | type | maxLength | Mask | Fill method |
|---|---|---|---|---|
| `edit-name` | text | 255 | none | `fill` |
| `edit-date-of-birth` | text | 255 | none | `fill` with `MM/DD/YYYY` |
| `edit-home-address` | text | 255 | none | `fill` |
| `edit-mailing-address-if-different-from-home-address-` | text | 255 | none | `fill`, or leave empty |
| `edit-mobile` | tel | 128 | none | `fill` with `NNN-NNN-NNNN` |
| `edit-email` | email | 254 | none | `fill` |
| `edit-if-yes` | text | 255 | none | `fill`, or leave empty |
| `edit-what-is-your-preferred-language` | select | — | — | `select` by option text |
| `edit-please-choose-the-wic-clinic-closest-to-you` | select | — | — | `select` by option text |
| `edit-can-you-receive-text-messages-yes` / `-no` | radio | — | — | `check` |
| `edit-do-you-have-medical-yes` / `-no` / `-in-progress` | radio | — | — | `check` |
| `edit-please-select-all-that-apply-*` | checkbox | — | — | `check` |
| `edit-i-authorize-my-wic-appointments-select-all-that-apply-*` | checkbox | — | — | `check` |
| `edit-url` | text | 255 | — | LEAVE EMPTY. Honeypot. |

### The Date Field Accepts a Plain Fill

`#edit-date-of-birth` is `type=text` with `maxlength 255`. It has no mask and no date
picker that captures keys. Use `fill` with the `MM/DD/YYYY` format. The readback gives
back the same string. The per-key `K` helper is not necessary on this field.

## Gates and Their Polarity

This form has no master gate. No answer hides or shows another block. Fill the fields
in any sequence.

Two radio groups are true radio groups, and not independent checkbox pairs. To set one
member clears the other members of the group. Read EVERY member of the group again
after the write, to confirm this.

| Group | Members | Effect |
|---|---|---|
| Can you receive text messages | `-yes`, `-no` | To set one clears the other. |
| Do you have MediCal | `-yes`, `-no`, `-in-progress` | To set one clears the other two. |

`#edit-if-yes` is the "If yes - MediCal Case #" field. The label makes it conditional,
but the field stays visible and enabled when the MediCal answer is No. The field does
not become disabled. Leave it empty when the answer is No. A CalWORKs id is NOT a
Medi-Cal case number. Do not put a CalWORKs id in this field.

The two checkbox groups are independent checkboxes with "select all that apply"
labels. Set only the members that the participant confirms. Leave the other members
clear.

- WIC category: `-pregnant`, `-post-partum`, `-infant-breastfeeding`,
  `-infant-formula`, `-childrentoddler-0-5`
- Appointment authorization: `-in-person`, `-virtual-phone`, `-telehealth-video`

## Mailing Address — Leave Empty When It Is the Same

The label of `#edit-mailing-address-if-different-from-home-address-` is "Mailing
address (if different from home address)". When the mailing address is the same as the
home address, an empty field is correct. Do not copy the home address into this field.

The address is ONE free-text line. The form has no separate city, state, or ZIP field.
Put the full address in one string.

## Exact Select Option Text

`#edit-what-is-your-preferred-language`:

```
- None -
English
Spanish
Other
```

`#edit-please-choose-the-wic-clinic-closest-to-you`:

```
Arlanza Riverside WIC      <-- this option is DUPLICATED, see the warning
Arlanza Riverside WIC      <-- duplicate
Banning WIC
Blythe WIC
Cathedral City WIC
Corona WIC
Desert Hot Springs WIC
Hemet WIC
Indio WIC
Jurupa WIC
Lakeshore WIC
Mecca WIC
Moreno Valley WIC
Riverside Neighborhood WIC
North Riverside WIC
Palm Springs WIC
Perris WIC
Rubidoux WIC
Temecula WIC
```

**WARNING: "Arlanza Riverside WIC" is in the list TWICE.** This is a defect of the
site. Select by option TEXT, and never by index. An index that a count gives can move
to the wrong option because of the duplicate.

`get value` on a select returns the option INDEX, and not the text. To read a select
again, use:

```bash
agent-browser get text "#edit-what-is-your-preferred-language option:checked"
agent-browser get text "#edit-please-choose-the-wic-clinic-closest-to-you option:checked"
```

## The Honeypot Field

`#edit-url` has the label "Leave this field blank". It is a spam honeypot. LEAVE IT
EMPTY. A value in this field identifies the submission as a bot submission, and the
site can reject the application without a message.

## The CAPTCHA

The form has a Google reCAPTCHA v2 checkbox.

- Sitekey: `6LdjsCcdAAAAAEZAN7mpCoyarPt_fZCpcJXlIGhk`
- Container: `.g-recaptcha`, with `data-sitekey`
- Token target: `#g-recaptcha-response`, a textarea. It is present but NOT visible.
  This is normal for reCAPTCHA v2.
- The visible checkbox is `iframe[src*='recaptcha/api2/anchor']`, with `size=normal`
  and `type=image`. The challenge popup is `iframe[src*='bframe']`.

An empty `get value "#g-recaptcha-response"` means there is no token, so nobody solved
the challenge.

Do not solve the challenge. Do not bypass the challenge. Do not script the challenge.
Report the state only:

```bash
agent-browser get count "iframe"
agent-browser get value "#g-recaptcha-response"
agent-browser is visible "iframe[src*='recaptcha/api2/anchor']"
```

## The Submit Condition

`#edit-submit` is a submit input with the value "Submit".

**`#edit-submit` is ENABLED when the form is empty, and it stays enabled when there is
no captcha token.** The button is never disabled. Do not use the state of the button
as proof that the form is complete, and do not use it as proof that the captcha is
complete. Drupal validates the captcha on the server after the submit step.

A human must solve the captcha before the submit step. Then get the approval of the
user. Then click `#edit-submit`.

## Screenshot Artifacts on This Page

Two elements are absent from a `screenshot --full` capture, but they ARE on the page.
Do not diagnose these as fill failures:

1. The sticky site header covers the Date of Birth row in the stitched full-page
   image. `get value "#edit-date-of-birth"` gives the correct value.
2. The reCAPTCHA widget does not show in the stitched image. The CAPTCHA heading
   shows, and the widget area looks empty. The anchor iframe is present and visible.

Confirm both with `get value` and `is visible`, and not with a screenshot.

## Payload Values With No Field on This Form

This form is short. The usual payload contains many values that have no field. Report
these as unused. Do not force them into a field that looks similar:

`record_id`, `file_open_date`, `dpss_referral_date`, `verbal_consent_provided`,
`funding_source`, `calworks_id`, `ethnicity`, `gender`, `special_needs`,
`farm_worker`, `marital_status`, the home phone, the work phone, the main phone,
`county`, the separate city, state, and ZIP fields (the address is one free-text
line), and the mailing address when it is the same as the home address.

## Warm-Path Fill

```bash
source .claude/skills/form-completion/scripts/fill-helpers.sh
SESSION=<session>

S fill "#edit-name" "<full name, with the middle name>"
S fill "#edit-date-of-birth" "<MM/DD/YYYY>"
S fill "#edit-home-address" "<street, city, state, ZIP on one line>"
S fill "#edit-mobile" "<NNN-NNN-NNNN>"
S fill "#edit-email" "<email>"
S select "#edit-what-is-your-preferred-language" "<English|Spanish|Other>"
S select "#edit-please-choose-the-wic-clinic-closest-to-you" "<exact clinic text>"
C "#edit-can-you-receive-text-messages-<yes|no>"
C "#edit-do-you-have-medical-<yes|no|in-progress>"
C "#edit-please-select-all-that-apply-<category>"
C "#edit-i-authorize-my-wic-appointments-select-all-that-apply-<mode>"
```

There is ONE Name field. Put the full name, INCLUDING the middle name, in it.

Readback. Do all of it. Read every member of the two radio groups:

```bash
V edit-name edit-date-of-birth edit-home-address edit-mobile edit-email
agent-browser get text "#edit-what-is-your-preferred-language option:checked"
agent-browser get text "#edit-please-choose-the-wic-clinic-closest-to-you option:checked"
agent-browser is checked "#edit-can-you-receive-text-messages-yes"
agent-browser is checked "#edit-can-you-receive-text-messages-no"
agent-browser is checked "#edit-do-you-have-medical-yes"
agent-browser is checked "#edit-do-you-have-medical-no"
agent-browser is checked "#edit-do-you-have-medical-in-progress"
```

## Exceptions Solved in the Cold-Start Session

- No field on this form needed the per-key `K` helper. Every `fill` was correct on the
  first attempt, and the date field also was correct.
- The empty widget area in the full-page screenshot is not a defect of the captcha.
  The anchor iframe is present and visible.
- The empty Date of Birth row in the full-page screenshot is not a fill failure. The
  sticky header covers the row.
