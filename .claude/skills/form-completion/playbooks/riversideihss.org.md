# Playbook: riversideihss.org (Riverside County IHSS Application)

Confirmed 2026-08-05 with agent-browser 0.33.2.

**WARNING: This is a live government benefits form. Do not submit test data. Do not
submit without explicit approval from the user.**

**Freshness probe.** Each command below must return 1. If a command does not return 1,
the site changed. Use the cold-start procedure. Write this file again.

```bash
agent-browser get count "#firstNameTxt"
agent-browser get count "#ssnTxt"
agent-browser get count "#btnSubmit"
```

## URL — Open the Form Directly

```
/Home/IHSSApply      landing page, 0 application fields ("Click here to apply online")
/Home/IHSSIntakeApp  second page,  0 application fields ("Click here to continue")
/IntakeApp           THE FORM:     128 inputs, 9 selects
```

Open `https://riversideihss.org/IntakeApp` directly. All the inputs on the first two
pages belong to Google Translate (`goog-gt-*`). Snapshots of this site contain 60 to 98
percent Google Translate lines (249 of 255 lines on the landing page). Use `#id`
selectors. Do not use snapshots here.

## Required Fields That the Usual Payload Does Not Contain — Ask First, in One Batch

- SSN (`ssnTxt`). A CalWORKs id or a record id is not an SSN.
- Living arrangement: `chkBxLIndependent`, `chkBxLAnyLFacility`, `chkBxLBoard`,
  `chkBxLHome`.
- Health history: six `chkBxHealthHistory` checkboxes.
- IHSS service need: `chkBxIHSSService` checkboxes and `chkBxOther`.
- SSI/SSP: `chkBxSSIYes` / `chkBxSSINo`.
- Assistance available at home: `chkBxAssistanceYes` / `chkBxAssistanceNo`.
- Lives alone: `chkBxLiveAloneYes` / `chkBxLiveAloneNo`.

Also ask these path decisions: self or on-behalf (the master gate), mailing address the
same as residential, real data or test data.

## Fill Sequence — Answer the Master Gate First

Set `chkBxApplyYourselfYes` when the applicant applies for the applicant. The
representative block stays hidden. Do not fill it.

Set `chkBxApplyYourselfNo` for an on-behalf or DPSS referral. This shows the
representative block: `repFirstNameTxt`, `repLastNameTxt`, `repPhoneTxt` (masked, ten
digits), `repEmailTxt`, `relToApplicantDrpDwn`, and the consent pair
`chkBxApplicantAgreeYes` / `chkBxApplicantAgreeNo`.

If you do not answer this gate, the consent checkbox stays hidden. A click on the
hidden checkbox shows success but does nothing.

The field `ticketId2Txt` stays hidden, also when the relationship is ASD Staff. Do not
try to fill it.

## Field Table (Section 1 and Confirmed Sections)

| Field | id | Method | Notes |
|---|---|---|---|
| First/Last name | `firstNameTxt` / `lastNameTxt` | fill | max 25 |
| Street + unit | `streetTxt` | fill | max 30, no separate unit field |
| City | `cityTxt` | fill | max 25 |
| State | `stateTxt` | **per-key** | masked, max 2 — use `CA`, not `California` |
| Zip | `zipCodeTxt` | fill | max 10 |
| SSN | `ssnTxt` | **per-key** | masked, 9 digits → `123-45-6789` |
| Birthdate | `birthDateTxt` | **per-key** | masked, `MMDDYYYY` → `01/02/2000` |
| Phone | `telephoneTxt` | **per-key** | masked, 10 digits |
| Email | `emailTxt` | fill | max 50 |
| Sex | `chkBxSexMale` / `chkBxSexFemale` | check | |
| Adopted minor | `chkBxAdoptedChildYes/No` | check | |
| Mailing same? | `chkBxMailAddressYes/No` | check | **"Yes" hides the mailing block. This is correct.** "No" shows `mailStreetTxt`, `mailCityTxt`, `mailStateTxt` (per-key), `mailZipCodeTxt` |
| Veteran | `chkBxVeteranYes/No` | check | "Yes" shows `veteranNameTxt`, `veteranClaimNumberTxt` |
| Health history | `healthHistoryTxt` | fill | max 125, required |
| Other service | `otherServiceRequestedTxt` | fill | starts disabled, no accessible name — `check "#chkBxOther"` enables it |
| Past IHSS | `chkBxPastIHSSYes/No` | check | "Yes" shows `pastIHSSDateTxt`, `pastIHSSCountyTxt`, `monthlyHoursTxt` |
| Household member | `relationshipDrpDwn`, `nameHouseholdTxt`, `birthDateHouseholdTxt`, `ssnHouseholdTxt` | select/fill | a participant_type of "Parent" goes here, not in the applicant fields. The DOB and SSN labels say "(optional)". A participant_type alone is not a household member — it gives no name. Leave the whole block empty unless the user gives a name. Do not invent a person. |
| Ethnicity | `ethnicDrpDwn` | select | **use "Hispanic" — "Hispanic/Latino" is not an option** |
| Languages | `languagePrepareToReadDrpDwn` / `languagePrepareToSpeakDrpDwn` | select | Speak id is `languagePrepareToSpeakDrpDwn`, **not** `languageSpeakDrpDwn` (confirmed 2026-08-05). Spanish has two options (NOA in English / NOA in Spanish) — select with care |
| Household DOB / SSN | `birthDateHouseholdTxt` / `ssnHouseholdTxt` | **per-key** | masked, same as the applicant equivalents |
| Blind / vision | `chkBxBlindYes/No`, `chkBxVisImpairedYes/No` | check | "Yes" shows the related accommodation checkboxes |
| Name used | `nameUsedTxt` | fill | optional, no `*` marker |
| Gender identity | `genderIdentityDrpDwn` | select | separate from the sex checkboxes; Another gender identity / Decline to state / Female / Male / Non-Binary (neither male nor female) / Transgender: female to male / Transgender: male to female |
| Sexual orientation | `sexualOrientationDrpDwn` | select | Another sexual orientation / Bisexual / Decline to state / Gay or lesbian / Queer / Straight/heterosexual / Unknown |
| Sex at birth | `chkBxSexOrigMale` / `chkBxSexOrigFemale` | check | a SECOND sex pair, separate from `chkBxSexMale/Female` |
| Veteran relative | `chkBxVeteranRelYes/No` | check | separate from `chkBxVeteranYes/No` |
| Receives IHSS now | `chkBxRcvIHSSServicesYes/No` | check | separate from `chkBxPastIHSSYes/No` |

### The Masks Add Their Own Format — Compare the Digits, Not the String

Send digits only with the per-key method. The mask inserts the separators. The readback
shows the formatted string, so a digit-for-digit comparison is the correct check
(confirmed 2026-08-05):

| Field | Keys you send | Value that `get value` returns |
|---|---|---|
| `ssnTxt` | nine digits | `123-45-6789` |
| `birthDateTxt` | MMDDYYYY | `01/02/2000` |
| `telephoneTxt` | ten digits | `(777) 777-7777` |
| `stateTxt` | the 2-letter state code | the same two characters, no separator |

A readback that differs from the keys you sent is not a failure here. A readback that
shows `__/__/____` or an empty string is a failure.

### There Are No `required` Attributes

`get count "[required]"` and `get count "[aria-required='true']"` both return 0. The site
marks the required fields with `*` in the label text and enforces them in JavaScript.
Do not conclude that the form has no required fields.

## No Middle-Name Field in Section 1

The applicant block has `firstNameTxt` and `lastNameTxt` only. There is no
`middleNameTxt`. A middle name in the payload has nowhere to go. Ask the user whether to
append it to the first or the last name. Do not silently drop it.

## IHSS Service-Need Checkboxes Share One id

The four service checkboxes all carry `id="chkBxIHSSService"` (invalid HTML on the site).
`get count "#chkBxIHSSService"` returns 4 and `#chkBxIHSSService` addresses only the
first. Use the `value` attribute instead — these are stable:

| value | Service |
|---|---|
| 80 | Domestic Services — household cleaning, meal preparation, laundry, food shopping |
| 81 | Personal Care — bathing, bowel/bladder, dressing, feeding, grooming |
| 82 | Transportation — medical appointments and health-related services |
| 83 | Paramedical Care |
| 84 | Other (separate id `#chkBxOther`; enables `otherServiceRequestedTxt`) |

```bash
agent-browser check "input[value='81']"
agent-browser is checked "input[value='81']"
```

The `name` attribute on these is empty, so `[name=…]` does not work. `nth-of-type` also
fails on this form because the matching inputs have different parents — it is a selector
artifact, not a missing element.

## Health-History Checkboxes Also Share One id

Same pattern as the service checkboxes: six inputs all carry `id="chkBxHealthHistory"`.
Address them by `value` (confirmed 2026-08-05). These are specific medical conditions,
not a general history:

| value | Condition |
|---|---|
| 95 | Unable to perform some activities of daily living (bathing, feeding, dressing, walking) |
| 96 | Currently under hospice care |
| 97 | Have a terminal illness likely to result in imminent death |
| 98 | Have a pending/recent organ transplant |
| 100 | Require around the clock use of supplemental oxygen |
| 101 | Have cancer, currently being treated |

Note the gap: there is no value 99. Do not scan value ranges with `get count` — the
`FIELDS` helper in `scripts/fill-helpers.sh` gives the full value and label map for
every group in one tool call.

```bash
agent-browser check "input[value='95']"
```

Do not put `[id='chkBxHealthHistory']` in the selector. A selector with `[id=…]`
resolves to the FIRST element with that id only. The compound form works for value 95
by accident and fails for 96 and above. Use `input[value='N']` alone, or check by the
visible label: `find role checkbox check --name "Currently under hospice care"`.

The separate free-text `#healthHistoryTxt` ("Briefly describe your health history") is
required and independent of these boxes.

## Read a Select With `option:checked`, Not `get value`

`get value` on a select returns the option **index**, which cannot be checked against the
intended text. Use the child selector instead (confirmed 2026-08-05):

```bash
agent-browser get text "#ethnicDrpDwn option:checked"   # -> "Hispanic"
```

Element-scoped `screenshot "#someSelect" out.png` captured an unrelated page region on
this form — do not use it to verify a select.

## Enumerating Labels Without `eval`

`get attr` does not accept XPath — an XPath selector returns `✓ Done` with no value,
which looks like success. To map same-id checkboxes to their human labels, use
`screenshot out.png --annotate --full` and read the `[N] @eN role "name"` legend. Grepping
that legend for `\*` also lists every required-marked field on the page.

## The Affirmation Expander Toggles — Check the State, Do Not Count Clicks

`click "div.header"` toggles. The section may already be open. One click can close
it and leave `#btnSubmit` disabled, which reads like a different bug. Read the state after
each click:

- `"+ Expand : Please expand and read below information…"` → closed, submit disabled
- `"- Collapse : Thank you for reading the important information."` → open, submit enabled

**`get text` can return a stale string for one call after the click.** Observed
2026-08-05: immediately after `click "div.header"`, `get text "div.header"` still returned
the `+ Expand` string and `is enabled "#btnSubmit"` returned `false`, but
`get html "div.header"` returned the `- Collapse` string in the same moment. One more read
of both selectors then agreed on `- Collapse` and `enabled = true`.

Use `get html "div.header"` for the authoritative state. If `get text` and `get html`
disagree, read again before you start a diagnosis. Also confirm with the body element:

```bash
agent-browser get html "div.header"        # authoritative: "- Collapse : Thank you..."
agent-browser is visible "div.content"     # true when the section is open
agent-browser is enabled "#btnSubmit"
```

Note that `div.header span` returns count 0 — the header holds a bare text node. An
earlier version of this file used `div.header.span`, which is a class selector and not a
descendant selector. Use `div.header`.

Select options (exact text):

- `relToApplicantDrpDwn`: Adult Services Division (ASD) Staff / Community Agency / Family Member / Health Plan Provider / Other
- `relationshipDrpDwn`: Child / Non-Relative / Other Relative / Parent / Spouse
- `ethnicDrpDwn`: American Indian or Alaskan Native / Asian Indian / Black / Cambodian / Chinese / Filipino / Guamanian / Hawaiian / Hispanic / Japanese / Korean / Laotian / Mixed Ethnicity / Other / Other Asian or Pacific Islander / Samoan / Vietnamese / White

## Section 9 and Submit

Open the affirmation section with a click, then read the text back to confirm state:

```bash
agent-browser click "div.header"       # The element has no id. The class selector is stable.
agent-browser get text "div.header"    # Expect "- Collapse : Thank you..."
agent-browser is enabled "#btnSubmit"
```

Use `div.header`, **not** `div.header.span` (corrected 2026-08-05). Both match one
element, but `get text "div.header.span"` returned the stale "+ Expand" string after the
section was already open, which reads like a failed click. `div.header` reports the live
text. Do not count clicks — the click toggles, so a second click closes the section again.

The section closes on page reload. Reopen it after any navigation.

### Turnstile — Read the Token, Do Not Count Iframes

Cloudflare Turnstile is on this page. The widget div is present and the site key is
`0x4AAAAAAAzaiJRgbzD5L2iV`:

```bash
agent-browser get count ".cf-turnstile"                        # 1 — widget div present
agent-browser get count "iframe"                               # 0 — see below
agent-browser get value "input[name='cf-turnstile-response']"  # the token, or "" 
```

**`get count "iframe"` returns 0 even when the challenge passed.** Confirmed 2026-08-05 on
local Chrome: `iframe` count was 0 and `cf-turnstile-response` held a full token
(approximately 700 characters), and `#btnSubmit` was enabled. The widget puts its iframe in
a place that this selector does not reach. An iframe count of 0 proves nothing.

**The token field is the only reliable signal.** A non-empty `cf-turnstile-response` means
the challenge passed. An earlier version of this file recorded `iframes=0` plus an empty
token as proof that the challenge never runs in this environment. The iframe half of that
test is wrong; only the empty token is meaningful.

If the token is empty and `#btnSubmit` stays disabled after the affirmation section is
open, wait and read the token again — it arrives asynchronously. Do not attempt to bypass
it. If it stays empty, stop and tell the user.

**Then stop. Get approval from the user before the submit step.**

## The Warm Path Is One Write Batch

Confirmed 2026-08-05 for a self-application: one Bash call held every write (the master
gate, ten applicant fields, the sex checkbox, the mailing gate, the three Yes/No pairs, the
living arrangement, the health-history box and free text, the four service boxes, and the
three selects). One readback pass then confirmed all of them. Every field was correct on
the first pass. No field needed a second write, and no gate hid a field that the batch
tried to fill, because the two gates (`chkBxApplyYourselfYes`, `chkBxMailAddressYes`) only
hide blocks that a self-application leaves empty.

Order the batch so the master gate comes first. Keep the per-key method for `stateTxt`,
`ssnTxt`, `birthDateTxt`, and `telephoneTxt`.

The unanswered Yes/No pairs stay clear, and both boxes read `false`. This is the correct
result for a value that the user did not give. Report the field as empty.
