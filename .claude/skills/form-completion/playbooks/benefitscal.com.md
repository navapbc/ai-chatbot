# Playbook: benefitscal.com (California CalFresh/SNAP Application)

Scout survey and eight fill-agent passes, 2026-08-05, with agent-browser 0.33.2.
Cold start. The fill agent completed the "Your Information" and "People"
sections, entered "Household Details", and stopped on the work-program detail
page `ABDWP` (page 41 of the chain below). Each stop before that was on a
forbidden data category. Everything after `ABDWP` is UNCONFIRMED.

**WARNING: This is a live government benefits form. Do not submit test data. Do not
submit without explicit approval from the user.**

**Freshness probe.** Each command below must return 1. If a command does not return 1,
the site changed. Use the cold-start procedure. Write this file again.

```bash
agent-browser get count "#primarylang"
agent-browser get count "#addressLine1"
agent-browser get count "#zip5"
```

## URL Chain — Forty-One Pages Confirmed So Far

```
 1. https://benefitscal.com/                                        home page
 2. .../ApplyForBenefits/begin/ABOVR?lang=en                         interstitial: "Here's how it works"
 3. .../ApplyForBenefits/ABHLT                                       "Helpful Tips" — has an account offer, and a no-account path
 4. .../ApplyForBenefits/ABDEI                                       "Diversity, Equity, and Inclusion Statement" — text only
 5. .../ApplyForBenefits/ABSNC                                       optional mood check — none of the choices are required
 6. .../ApplyForBenefits/ABNAV                                       "Application Summary" — the master section list
 7. .../ApplyForBenefits/ABLPR                                       "Language Preferences"
 8. .../ApplyForBenefits/ABNMI                                       "Name Information"
 9. .../ApplyForBenefits/ABNHA                                       "Home Address" — has a gate, an address-check modal, a county picker
10. .../ApplyForBenefits/ABMAD                                       "Mailing Address" — gate: same as home?
11. .../ApplyForBenefits/ABCON                                       "Contact Information" — phone and email
12. .../ApplyForBenefits/ABCOP                                       "Contact Preferences" — text/email alert opt-in
13. .../ApplyForBenefits/ABPRI                                       "Select Benefit Programs" — narrows the rest of the flow
14. .../ApplyForBenefits/ABCSD                                       "CalFresh Submit App Divider" — an informational nudge page
15. .../ApplyForBenefits/ABDIS                                       "Disability" — a forbidden data category, answered after the user confirmed
16. .../ApplyForBenefits/ABCOS                                       "College Student" — a forbidden data category, answered after the user confirmed
17. .../ApplyForBenefits/ABCFA                                       "CalFresh Authorized Rep" — "authorize someone to help with your case?"
18. .../ApplyForBenefits/ABCFS                                       "CalFresh Spend Benefits" — "name someone to get/spend your benefits?"
19. .../ApplyForBenefits/ABRDT                                       "Birthdate" — a MASKED field, see below
20. .../ApplyForBenefits/ABSSN                                       "SSN" — a three-way status gate, answered after the user confirmed
21. .../ApplyForBenefits/ABSNA                                       "SSN A" — the actual SSN digit-entry page, a MASKED field
22. .../ApplyForBenefits/ABMRS                                       "Marital Status" — a radio group of 8 options
23. .../ApplyForBenefits/ABCID                                       "Citizenship Immigration Divider" — an informational nudge page, no fields
24. .../ApplyForBenefits/ABDOC                                       page title "Citizenship" — the URL slug (ABDOC) does not match the page title or topic
25. .../ApplyForBenefits/ABBID                                       "Background Info Divider" — no fields
26. .../ApplyForBenefits/ABASX                                       "Assigned Sex" — "What's your gender?"
27. .../ApplyForBenefits/ABGNR                                       "Gender" — "What's your gender identity?" (a second, separate gender page)
28. .../ApplyForBenefits/ABSXO                                       "Sexual Orientation" — optional, Next works with no answer
29. .../ApplyForBenefits/ABHSP                                       "Hispanic" — Hispanic/Latino/Spanish origin, a Yes/No/prefer-not question
30. .../ApplyForBenefits/ABRAE                                       "Race and Ethnicity" — a race select, separate from ABHSP
31. .../ApplyForBenefits/ABYSD                                       "Your Information Section Completed" — milestone, click "START THE NEXT SECTION"
32. .../ApplyForBenefits/ABNAV                                       "Application Summary" again — click "Start People"
33. .../ApplyForBenefits/ABHSD                                       "Household Member Definition" — other people in the household?
34. .../ApplyForBenefits/ABPLS                                       "People Summary" — a read-only list of the household members
35. .../ApplyForBenefits/ABTCD                                       "People Section Completed" — milestone
36. .../ApplyForBenefits/ABNAV                                       "Application Summary" again — click "Start Household"
37. .../ApplyForBenefits/ABHGW                                       "Household Gateway" — a situational checklist
38. .../SupportRequest/ABDEG                                         "ABAWD Exemption Screener" — a checklist. The URL path changes to /SupportRequest/ from this page on.
39. .../SupportRequest/ABDWR                                         "Work Requirement Screener" — work/volunteer/training checklist, a BRANCH GATE
40. .../SupportRequest/ABDWT                                         "Work Or Training Info" — hours for each week (opens when ABDWR has a work/training check)
41. .../SupportRequest/ABDWP                                         "Work Or Training Info" — organization name — LAST CONFIRMED PAGE
```

**Do not trust the URL slug as the page topic.** `ABDOC` asks about citizenship,
not "documents." Read the page title and heading to identify a page, not its
URL segment.

Click through in this order: APPLY FOR BENEFITS -> BEGIN -> Next -> Next -> Next
(without a mood choice) -> Start Your Information -> Next through ABLPR, ABNMI,
ABNHA (plus its modal and dialog, see below), ABMAD, ABCON, ABCOP, ABPRI -> on
ABCSD click "CONTINUE APPLICATION", **not** the "Skip and submit now (not
recommended)" link -> Next through ABDIS, ABCOS, ABCFA, ABCFS, ABRDT.

**Ids on this site are sometimes generic and do not name the question they
belong to.** Confirmed on `ABCFA` (id `filingTax` for a "authorize a
representative" question — nothing to do with tax filing) and `ABCFS` (id
`select_group` for a "name someone to spend your benefits" question). Do not
guess a field's purpose from its id. Always confirm the Yes/No mapping by
reading the visible label text, not by id name or by position.

**Two different markup patterns carry the label text — check both.** Most
radio pairs use `<label for="<id>">`. Confirmed on `ABSSN`: that page instead
uses `aria-labelledby` pointing at a separate `<span id="..._inner_label">`
with the visible text. A selector that only checks `for=` can miss the label
on a page built this way. If a `for=` lookup finds nothing, check
`aria-labelledby` and its target span next before you conclude the page has no
accessible label.

The page advance button on these pages has no id. Match it by its visible text,
"Next" (or "CONTINUE APPLICATION" on `ABCSD`).

**Back navigation: the "Back to Previous Page" button is safe, the step-tracker
links are NOT.** A fill agent went back five pages (from `ABDIS` through `ABNMI`)
with the site's own "Back to Previous Page" button to fix a field, then
re-advanced through every page again. Every value it had entered on the pages in
between was still there. But a jump through a step-tracker link (example: "Your
Information Reviewed") RESET corrected fields to their earlier values — a
corrected last-name/suffix split reverted to the combined value, and the suffix
select reverted to "-Select One-". Use "Back to Previous Page" for corrections.
After any use of a review link, re-verify every field that was corrected earlier.

**Some ids regenerate on every page load.** Fields on several pages (`ABRAE`,
`ABDEG`, `ABDWR`, `ABDWT`) carry a generated id with the pattern
`lift-ux-id-<32 hex chars>` or `..._N`. The id changes on each load — do not
save one, do not reuse one across visits, and do not put one in a freshness
probe. The `model` attribute on the same element is constant (example:
`model="ABRAE.APP_INDV_Collection.race_cd"`) — find these fields through the
`model` attribute in the page HTML, then use the id from THAT load. The pattern
is not universal: other fields on the same pages keep static ids (example:
`#ABDWP_PrgmName1`).

**A partially completed section restarts from its first page.** Clicking "Start
Household" on `ABNAV` when the section shows as not complete opened the section's
FIRST page (`ABHGW`), not the page where the previous pass stopped. There is no
mid-section resume. Expect to walk and re-verify every page of a section that a
previous pass left incomplete.

**The step counter changes when the program selection narrows.** Before
`ABPRI`, pages show "Step 1 of 9". After a fill agent chose CalFresh only on
`ABPRI`, the counter became "Step 1 of 8" from `ABCSD` onward. Do not treat a
changed step count as a sign of a broken fill — read it as a sign of which
programs are selected.

## Account Requirement — None, if You Take the Right Path

The "Helpful Tips" page (`ABHLT`) offers a "Create an account now" link. IGNORE it.
The same page has a "Next" button that continues WITHOUT an account. Confirmed
2026-08-05: the scout reached the first form page with no account and no login.

## Section List (From the Application Summary Page, `ABNAV`)

This page is the master progress list for the whole application. Section 2 was
disabled at scout time (labeled "People Not Available") until Section 1 has data
in it.

| # | Section | Status |
|---|---|---|
| 1 | Your Information | CONFIRMED — completed in one run, pages 7 to 31 of the chain |
| 2 | People | CONFIRMED — pages 33 to 35. Only the "no other people" path is confirmed. |
| 3 | Household Details | PARTIAL — pages 37 to 41. The section restarts from `ABHGW` when it is not complete. |
| 4 | Income | UNCONFIRMED — locked ("Not Available") until Household Details is complete |
| 5 | Expenses | UNCONFIRMED |
| 6 | Assets | UNCONFIRMED |
| 7 | Other Situations | UNCONFIRMED |
| 8 | Document Upload | UNCONFIRMED |
| 9 | Review & Submit | UNCONFIRMED |

The first form page (`ABLPR`) also shows its own "Step 1 of 9" sub-stepper inside
Section 1. Steps 2 to 9 of this inner sequence are also UNCONFIRMED.

## Field Table — Page `ABLPR` ("Language Preferences")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `primarylang` | select | What language do you prefer to read? | required | select by exact text | Shows a "-Select One-" placeholder, but `get value` reads `03` (English) as the current selection at page load — confirm with `option:checked` before you trust a default |
| `spokenlang` | select | What language do you prefer to speak? | required | select by exact text | Same option list as `primarylang`, plus "American Sign Language" |
| `applang` | select | In what language would you like to complete this application? | required | select by exact text | Pre-selected to English (value `03`), no "-Select One-" placeholder. Shorter option list than the other two selects — check the option text is in THIS list before you select |

No text `input` fields are on this page (`get count input` returned 0). One
`iframe` is present: an "Ask Robin" chat-assistant widget
(`daghqzwjxmect.cloudfront.net`) — not a form field, do not enter data in it. A
"Need Language Help?" accordion is present with no additional fields observed.

Confirmed by a fill agent: when the record language is English, all three selects
already default to English. No write is needed — read each with
`get text "#id option:checked"` to confirm, then move on.

No county selector is on this page.

## Field Table — Page `ABNMI` ("Name Information")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `text1` | text | First Name | required | fill | |
| `text2` | text | Middle Name | optional | fill | |
| `text3` | text | Last Name | required | fill | See the suffix note below before you put a suffix word here |
| `suffix` | select | Suffix | optional | select by exact text | Options: I, II, III, IV, V, VI, VII, VIII, IX, X, Jr., Sr. |
| `text4` | text | Other Names | optional | fill, or leave empty | |

**A suffix such as "II" has its own dropdown. Do not append it to the last name.**
A fill on this site put a last name and its suffix word whole into `text3`
(example: "Smith II") because the source value carried the suffix as part of the
last name. The site has a `suffix` select with a matching "II" option. Split a
last-name value that ends in a suffix word (II, III, Jr., Sr., and so on) between
`text3` and `suffix`.

The ids `text1` to `text4` are generic and not name-derived. The `FIELDS` helper's
HTML-based labels can misread them, because this page's markup puts placeholder
text before the real label. Cross-reference with `snapshot -i -u` for the true
label of each box on this page.

## Field Table — Page `ABNHA` ("Home Address")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `radioCard_0` / `radioCard_1` | radio | Are you experiencing homelessness? | appears required, a gate | check | Yes = `radioCard_0`, No = `radioCard_1`. This is a GAP-DECISION — the usual payload has no housing-status value. Do not derive it from an address on file (see `references/gap-analysis-and-provenance.md`, "Values You Must Not Derive"). Ask the user. |
| `addressLine1` | text | Address Line 1 | required | fill | |
| `addressLine2` | text | Address Line 2 | optional | fill | |
| `city` | text | City | required | fill | |
| `state` | select | State | required | none — DISABLED | Pre-set to California and locked. This site serves California only. Do not try to change it. |
| `zip5` | text | Zip Code | required | fill | |

**The homelessness gate sits above the address fields in the DOM, but does not
hide or disable them.** A fill agent filled every address field while the gate
was unanswered, and every fill succeeded. This is different from a hide-on-gate
site: read the gate's own answer requirement from the Next-button behavior, not
from whether the address fields accept a fill. Confirmed value mapping: No =
`radioCard_1`.

**`zip5` can read back empty right after the homelessness radio is checked, even
though the ZIP was correct moments before.** Confirmed once in one run — not
yet seen a second time. Re-read `zip5` right before you click Next, and refill
it if it is empty. Treat this as a possible site quirk on this page, not a
one-time fluke, until a second run confirms or clears it.

**An address-check modal can appear after Next on this page.** If the entered
address does not match the site's validation data (for example, a test
address), a modal opens: heading "We can't validate your address. Let's double
check if it's right.", a pre-checked radio "Address You Entered", a button "USE
THIS ADDRESS", and a link "Correct my address". Click "USE THIS ADDRESS" to
keep the address exactly as entered. Do NOT click "Correct my address" — that
lets the site rewrite the value the user gave.

**A county picker can appear after the address modal.** If the site cannot
derive the county from the address, a dialog opens: heading "What county are
you currently in?", a required `select#county`, and a "CONTINUE" button. Select
the county by exact option text, then verify with
`get text "#county option:checked"`.

**Both dialogs return on every visit.** A second pass through this page, with
the address unchanged and already accepted, opened the address modal and the
county dialog again. Expect both dialogs on EVERY Next click on this page, not
only the first.

**Confirmed stop point (first pass):** a fill agent stopped on this page with
the homelessness gate unanswered, and did not click Next. A second pass, after
the user answered No, passed through the modal and the county picker
successfully and continued to `ABMAD`.

## Field Table — Page `ABMAD` ("Mailing Address")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `mailadr1_radio_0` (Y) / `mailadr1_radio_1` (N) | radio | Do you get your mail at a different address? | gate, no default | check | **Inverse polarity: "No" means the mailing address is the SAME as home.** Selecting No shows no extra text block (confirmed). Selecting Yes is UNCONFIRMED — expected to reveal a mailing-address block, not yet tested. |

## Field Table — Page `ABCON` ("Contact Information")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `homePhone` | text | Home Phone | optional | fill, or leave empty | |
| `mobilePhone` | text | Mobile Phone | optional | fill | Ten digits go in; confirmed the site auto-formats to `(NNN) NNN-NNNN` on a plain `fill` — no mask helper needed here |
| `altPhone` | text | Work Phone/Alternate Phone | optional | fill, or leave empty | |
| `mail` | text | Email | optional, maxlength 100 | fill | This id is reused with a different role on `ABCOP` — see the warning below |

## Field Table — Page `ABCOP` ("Contact Preferences")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `mail` | checkbox | Email (alert opt-in) | optional | check, or leave clear | **Same id `mail` as the EMAIL TEXT FIELD on `ABCON` — a different page, a different element, a different role (checkbox here, not text).** Confirm which page you are on before you use `#mail`. |
| `phone` | checkbox | Text Message (alert opt-in) | optional | check, or leave clear | |
| `acceptance_agreement` | checkbox | Terms/Privacy acceptance | disabled by default | none — becomes enabled only when an alert box above it is checked (UNCONFIRMED which one) | |

Neither alert checkbox has an HTML `required` attribute, and the Next button
stayed enabled with both clear. This page's checkboxes are a contact
PREFERENCE, not a required field — the skill's "Values You Must Not Derive"
rule against inventing a contact preference still applies: leave both clear
unless the user states a preference.

## Field Table — Page `ABPRI` ("Select Benefit Programs")

| id | type | Label | Required | Fill method | Notes |
|---|---|---|---|---|---|
| `snap` | checkbox | Food (CalFresh) | part of a required group (pick at least one) | check | Independent of the other two — CalFresh alone is a valid, uncontested selection |
| `tanf` | checkbox | Cash Aid (CalWORKs, TCVAP, RCA) | part of the same group | check, or leave clear | |
| `medicaid` | checkbox | Health Coverage (Medi-Cal) | part of the same group | check, or leave clear | |
| `label_0` (Y) / `label_1` (N) | radio | Are you applying for benefits for yourself? | required | check | Confirmed: Yes = `label_0` |

**The program selection here narrows every page after `ABCSD`.** Selecting
CalFresh only changed the step counter from "Step 1 of 9" to "Step 1 of 8" from
`ABCSD` onward (see the URL-chain section). Expect a different page sequence
after this point if a future fill selects more than one program.

## Page `ABCSD` ("CalFresh Submit App Divider") — No Fields, One Choice of Button

An informational nudge page with no form fields. Two buttons: "CONTINUE
APPLICATION" (advances to the next page in the normal sequence) and "Skip and
submit now (not recommended)" (a shortcut toward submit). **Always use "CONTINUE
APPLICATION." Never use the skip link** — it is submit-adjacent and skips
sections the gap analysis has not covered.

## Field Table — Page `ABDIS` ("Disability")

| id | type | Label | Required (HTML) | Notes |
|---|---|---|---|---|
| `disability0` (Y) / `disability1` (N) | radio | Are you a person with a disability and need help to apply? | no `required` attribute found | FORBIDDEN DATA CATEGORY — do not answer from a derived guess. Ask the user. Confirmed value mapping: read the visible `<label for="disability1">No</label>` text before you click — do not assume index 0/1 maps to Yes/No. |
| `deaf0` (Y) / `deaf1` (N) | radio | Are you a person who is deaf or hard of hearing? | no `required` attribute found | Same rule as above. |

Answered No/No after the user confirmed both, verified by reading the label
text before the click and `is checked` on both ids of each pair after.

## Field Table — Page `ABCOS` ("College Student")

| id | type | Label | Required (HTML) | Notes |
|---|---|---|---|---|
| `CollegeStudentE_radio_button_0` (Y) / `CollegeStudentE_radio_button_1` (N) | radio | Are you a college student? | none found in HTML | FORBIDDEN DATA CATEGORY (student status) — ask the user. |

Confirmed stop, verified clean: neither radio was touched, Next was not
clicked. Answered No after the user confirmed it, and the fill continued.

## Field Table — Page `ABCFA` ("CalFresh Authorized Rep")

| id | type | Label | Notes |
|---|---|---|---|
| `filingTax_0` (Y) / `filingTax_1` (N) | radio | Do you want to authorize someone to help you with your CalFresh case? | The id name (`filingTax`) is unrelated to the question — do not trust it. Confirm the Yes/No mapping from the label text. Answered No; do not invent a representative. |

## Field Table — Page `ABCFS` ("CalFresh Spend Benefits")

| id | type | Label | Notes |
|---|---|---|---|
| `select_group_0` (Y) / `select_group_1` (N) | radio | Do you want to name someone to get and spend your CalFresh benefits for you? | Same generic-id warning as `ABCFA` — the id (`select_group`) does not name the question. Answered No; do not invent a person. |

## Field Table — Page `ABRDT` ("Birthdate") — a Masked Field

| id | type | Label | Required | Mask | Fill method | Notes |
|---|---|---|---|---|---|---|
| `birthDate_primary_input` | `type="password"`, `datatype="date"` | Date of Birth | required | YES | **per-key (`K`)** | Placeholder shows `MM/DD/YYYY`, attribute `maxdate="today"`. Confirmed: this input renders as a password-type field. A plain `fill` is expected to fail silently here, the same as the masked fields on other sites in this skill — use the keypress helper and read the value back with `get value` to confirm the formatted date. |

## Field Table — Page `ABSSN` ("SSN")

| id | type | Label | Notes |
|---|---|---|---|
| `ssn_group0` | radio | "Do you have a Social Security number?" — option Yes | GAP-DECISION, not a simple forbidden-category skip. Ask the user which of the three options is true. |
| `ssn_group1` | radio | same question — option No | |
| `ssn_group2` | radio | same question — option "I don't have it right now" | |

**This page is a three-way STATUS gate, not a text box for SSN digits.** It
does not ask for the number itself — it asks whether one exists. Do not treat
"SSN is forbidden data" as a reason to always skip this page with no answer:
the status question itself is answerable (the user can say whether the
applicant has a number), and the site's later flow likely depends on the
answer. Stop and ask the user to choose Yes / No / "I don't have it right now"
before you continue. If the answer is Yes, the site opens a SEPARATE digit-entry
page next (`ABSNA`, below) — treat entering the digits as its own decision,
even after the status gate is answered Yes.

Confirmed: choosing Yes on this gate opens `ABSNA` next.

## Field Table — Page `ABSNA` ("SSN A") — the Actual SSN Digit Entry, a Masked Field

| id | type | Label | Required | Mask | Fill method | Notes |
|---|---|---|---|---|---|---|
| `ssn` | `type="password"`, `inputmode="numeric"` | Social Security Number | not explicitly marked, but required for this flow | YES — masked | **per-key (`K`)** | Confirmed: nine digits sent by keypress read back as `NNN-NN-NNNN` (formatted, digits match). No confirm/re-enter box on this page — a single field only. A "Show social security number" toggle button is present but not needed; verify with `get value`, not by revealing the digits on screen. |

## Field Table — Page `ABMRS` ("Marital Status")

| id | type | Label | Notes |
|---|---|---|---|
| `maritalStatus_0` … `maritalStatus_7` | radio group (8 options) | What's your marital status? | Exact option order and text: Common Law, Divorced, Married, Never Married, Registered Domestic Partner, Separated, Single, Widowed. A payload value of "single parent household" matches "Single" — a direct, unambiguous match on the household-status word. Do not match it to "Never Married," which is a different, more specific legal status the payload does not state. |

## Page `ABCID` ("Citizenship Immigration Divider") — No Fields

An informational transition screen with no form fields, the same pattern as
`ABCSD`: text along the lines of "Next, let's go over some questions about
citizenship and immigration," then a `Next` button. **This site uses these
divider pages before it opens a new sensitive topic area.** Expect a divider
page like this before other sensitive sections too (income, assets) — a page
with no fields and a topic-announcement sentence is not a fill failure, it is
this site's own section-transition pattern.

## Field Table — Page `ABDOC` ("Citizenship")

| id | type | Label | Notes |
|---|---|---|---|
| `citizen_radio_0` (Y) | radio | Are you a U.S. citizen or national? — option Yes | FORBIDDEN DATA CATEGORY (citizenship/immigration status) — ask the user. |
| `citizen_radio_1` (N) | radio | same question — option No | |

The value mapping is confirmed through both label patterns on this page
(`<label for=...>` and the `_inner_label` span) — the two agreed. A Yes answer
opened no immigration follow-up page.

## Field Tables — the Demographic Pages (`ABASX`, `ABGNR`, `ABSXO`, `ABHSP`, `ABRAE`)

| Page | id | type | Label | Options |
|---|---|---|---|---|
| `ABASX` | `assignedSex_0` / `_1` / `_2` | radio | What's your gender? | Female / Male / I prefer not to answer |
| `ABGNR` | `gender_0` … `gender_6` | radio | What's your gender identity? | Another Gender Identity / Female / Transgender: Female to Male / Male / Transgender: Male to Female / Non Binary / I prefer not to answer |
| `ABSXO` | (radio group) | radio | Sexual orientation | Straight or Heterosexual / Gay or Lesbian / Bisexual / Queer / Another Sexual Orientation / Unknown / I prefer not to answer |
| `ABHSP` | `person_0` (Y) / `_1` (N) / `_2` | radio | Are you of Hispanic, Latino, or Spanish origin? | Yes / No / I prefer not to answer |
| `ABRAE` | dynamic id, `model="ABRAE.APP_INDV_Collection.race_cd"` | select | What is your race and ethnic origin? | -Select One-, American Indian or Alaskan Native, Asian, Black or African American, Native Hawaiian or Other Pacific Islander, Other or Mixed, White, I prefer not to answer |

- `ABASX` and `ABGNR` are two separate questions (assigned sex, then gender
  identity). A payload gender value answers `ABASX`. Ask the user before you
  answer `ABGNR` with the same value — the two facts can differ.
- `ABSXO` is optional: no `required` attribute, and Next works with no answer.
  Leave it with no answer when no source states one.
- **The site splits Hispanic origin from race** (the standard US census split).
  An ethnicity value such as "Hispanic/Latino" answers `ABHSP` (Yes), and does
  NOT map to any option in the `ABRAE` race list. `ABRAE` is optional — when no
  source states a race, leave it at "-Select One-" and put race in the gap
  analysis.

## Field Tables — the People Section (`ABHSD`, `ABPLS`)

| Page | id | type | Label | Notes |
|---|---|---|---|---|
| `ABHSD` | `hshld_radiogrp_0` (Y) / `_1` (N) | radio | Do you have other people living in your household? | The gate for the whole People section. A Yes answer is expected to open member-entry pages (UNCONFIRMED — only the No path is confirmed). |
| `ABPLS` | none | — | People Summary | Read-only. Lists each household member with age. An "ADD ANOTHER" button is present. |

## Field Tables — the Situational Checklists (`ABHGW`, `ABDEG`, `ABDWR`)

Three checklist pages follow the People section. Each shows a list of
situations plus a "None of these apply" checkbox.

| Page | ids | Items |
|---|---|---|
| `ABHGW` | `govtaid`, `disability`, `college`, `food`, `living`, `breastfeeding`, `military`, `none` | Received public assistance in any state / person with a disability / enrolled in college or trade school / get food from somewhere other than at home / live in a facility, shelter or other living arrangement / breastfeeding a child / serving or served in the U.S. Military (or a dependent) / None of these apply |
| `ABDEG` | dynamic ids `lift-ux-id-..._0` … `_11` | Health issue that makes work hard / personal issue that makes work hard / caring for a child under 14 / caring for a child under 6 / caring for a person who needs help / currently pregnant / in school at least half-time / getting or applied for unemployment benefits / getting or applied for disability benefits / Indian, Urban Indian, or Californian Indian / ORR Training Program at least half-time / None of these apply |
| `ABDWR` | dynamic ids `..._0` … `_3` | Working / community service or volunteer work / work, education, or training program / None of these apply |

Rules for these checklists:

- Check "None of these apply" ONLY when a confirmed answer rules out every
  listed item. When one item is not ruled out, stop and ask the user. On one
  run, "no income" ruled out "Working" but not volunteering or a training
  program — the agent stopped, and that was correct.
- **`ABDWR` is a branch gate.** A check on the work/education/training item
  opened a detail chain: `ABDWT` (hours for each week), then `ABDWP`
  (organization name). Expect more detail pages after those (program pay and
  start date are plausible, UNCONFIRMED). Warn the user about this chain in the
  gap analysis before a work-program item is checked.
- **The `ABDEG` checkbox inputs report `is visible` = false, and the checks
  still work.** The visible element is a styled sibling, not the input. Use
  `check` and `is checked` on the input id. Do not use `is visible` to validate
  this page. Two of the twelve inputs (`_2` "child under 14", `_6` "school
  half-time") are in the HTML but never appeared in the snapshot tree — read
  this page's items from `get html`, not from the snapshot alone.

## Field Tables — the Work-Program Detail Pages (`ABDWT`, `ABDWP`)

| Page | id | type | Label | maxlength | Fill method |
|---|---|---|---|---|---|
| `ABDWT` | dynamic id, `model="ABDWT.CpAbawdDetails_Collection.workHours"` | text | Total hours for each week | 3 | plain `fill` — no mask |
| `ABDWP` | `ABDWP_PrgmName1` (static) | text | Organization or Person's Name ("Where do you do your work program or education or training program?") | 60 | not filled — LAST CONFIRMED PAGE. A disabled "ADD ANOTHER" button is present (UNCONFIRMED — likely for a second program). |

## A Site-Wide Pattern: Individual-Status Gates Never Carry a `required` Attribute

Confirmed across four separate pages (`ABNHA` homelessness, `ABDIS`
disability and hearing, `ABCOS` college student, `ABCFA`/`ABCFS` third-party
authorization): a personal-status Yes/No question on this site never has an
HTML `required` attribute, and its Next button stays enabled even when the
question is unanswered. Do not read the absence of `required` or an enabled
Next button as "this question is optional." Every one of these questions is a
single-question page with two or three named radios (`..._0`/`..._1`, or a
word pair) — that page shape, by itself, is the site's own signal that the
question needs a real answer, HTML attribute or not. Later pages confirmed the
pattern: citizenship (`ABDOC`), the household gate (`ABHSD`), and the three
situational checklists all follow it. Expect more pages shaped like this in the
unconfirmed sections (veteran status and striker status are plausible). Ask the
user before you answer any of them; never leave one at a guessed default
because the button did not block you.

## Bot Checks

None seen across the forty-one pages reached so far (home page through
`ABDWP`). Confirm this again if a fill agent reaches later sections — a
captcha can appear later in a multi-page flow even when the entry pages have
none.

## Unreached Pages — Ask the User Before You Guess

Everything after `ABDWP` is UNCONFIRMED: the pages after the organization-name
question (program pay and program start date are plausible follow-ups), the
rest of Household Details, and the Income, Expenses, Assets, Other Situations,
Document Upload, and Review & Submit sections. Known question areas from the
gap answers of one run, with no confirmed pages yet: income sources, rent or
mortgage, utilities (the form can split them into categories — collect the
categories when it does), and savings. Financial account identifiers (bank
name, account number) stay forbidden — a form does not need them, and the gap
analysis must not ask for them. The People section's multi-member path is also
UNCONFIRMED (a household with more than one person needs a name and birth date
for each member — a common stop point when a payload describes only the primary
applicant). Expect 20 or more gaps for the full application. Do not derive a
household size or an income answer from an adjacent fact (see
`references/gap-analysis-and-provenance.md`, "Values You Must Not Derive").
Ask the user. At Review & Submit, re-verify the name fields — a step-tracker
review link can revert a corrected name split (see the back-navigation warning
above).

## Forbidden and Sensitive Categories Seen So Far

Some pages ask for data that a fill agent should not answer even when a value
is technically available or pattern-matchable, because a wrong or guessed
answer on a benefits application has consequences beyond a simple form error.
Confirmed categories seen through `ABDOC`: disability status, hearing status,
college-student status, citizenship/immigration status, and Social Security
number (the status gate is answerable by the user; the digits, on the
follow-up page `ABSNA`, are a separate and stricter stop — ask again even
after the status gate is answered Yes). Treat these the same way as financial
account numbers — ask the user, do not guess, do not pattern-match from an
adjacent fact. A value the user already gave for a DIFFERENT application in
the same session (example: an SSN typed for a different site's form) is not
automatically valid for this site — ask again before reusing it.
