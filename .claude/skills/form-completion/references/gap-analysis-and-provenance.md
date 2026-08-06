# Gap Analysis and Provenance

Two reports are mandatory in each form run. Show the gap analysis before you fill.
Show the provenance report after you fill. This file gives the format and the rules
for both.

## Part 1 — the Gap Analysis (Before the Fill)

Show the user a gap-analysis table before the first write. Do not fill a field before
the user answers all the required gaps. Put each field in one of these groups:

| Group | Meaning | Action |
|---|---|---|
| READY | The payload or the user gave the value. | Fill it. No question. |
| DERIVED | A rule changes the value to the form format. | Show the rule in the table. Fill it. |
| GAP-REQUIRED | The form requires the field. No source has the value. | Ask. Do not fill before the answer. |
| GAP-DECISION | The answer changes the fill procedure. | Ask. Examples: self or on-behalf, mailing address the same as residential. |
| NO-FIELD | The payload has the value. The form has no field for it. | Show it in the final report as unused. |

Example table to show the user:

```
GAP ANALYSIS — <the application name>
READY (12):      first name, last name, street, city, zip, date of birth, phone, ...
CHANGED (3):     "California" becomes "CA" — the form wants 2 letters
                 "Spanish (Mexico)" becomes "Spanish" — the form's own list
                 street and apartment go on one line — the form has one box
MISSING (6):     Social Security Number, living arrangement, income,
                 household size, health information, service needs
DECISIONS (2):   applying for herself or with help; mail goes to the same address?
NO PLACE (5):    middle name, work/cell phones, case numbers, ...
```

The table goes to the user. Use the plain group names above (CHANGED, MISSING,
DECISIONS, NO PLACE). Keep the technical group names (DERIVED, GAP-REQUIRED,
GAP-DECISION, NO-FIELD) in the transcript and in the agent reports.

### Rules for the Questions

- Ask all the gap questions in the same phase, before the fill.
- One `AskUserQuestion` call holds a maximum of four questions. If there are more
  than four gaps, use two or more calls in sequence. Do not remove gaps to fit one
  call. **Do not guess the answers that did not fit.**
- If the user does not give an answer for a required gap, keep the field empty. Show
  the empty field in the final report. An empty field is correct. An invented value
  is not correct.
- Treat all data as real data. Do not ask if the data is real or test data. The
  submit step always needs approval, and that approval is the protection.
- **Each session is a new participant.** Ask for each missing required value in each
  session. Do not keep participant answers between sessions. Do not write
  participant values or decisions to any file that lives after the session. Site
  facts go in the playbook. Participant data does not.

### Ask for the Value, Not for Instructions

A gap question collects data for one field. Ask for the missing value directly. Do
not ask the user what to do about a missing value.

Bad (a process question):

> The payload has no SSN. Required field `ssnTxt`. How should I handle it?
> 1. Leave empty, report it
> 2. Use a test SSN

Good (a data question):

> What is Maria's Social Security Number? You can type it, or choose an option.
> 1. Leave it empty for now
> 2. She does not have one

The user types the value through the free-text option of the question. Ask for the
value one time. If the user leaves it empty, keep the field empty and show it in
the report.

### Match the Question Type to the Field Type

| Field type | Question format |
|---|---|
| Typed value (SSN, name, date) | A direct question. Give two options: "Leave it empty for now" and one real alternative (example: "She does not have one"). The user types the value in the free-text option. |
| Binary (Yes/No, lives alone) | Two plain options. Example: "Yes, she lives alone" / "No, other people live there". |
| One choice from a list (living arrangement) | One option for each choice, in plain words. |
| Check-all-that-apply (health history, service needs) | Set `multiSelect: true`. One option for each checkbox. Add "None of these apply" when that is a valid answer. |

One question holds a maximum of four options. If a checkbox group has more than
four items, split the group across two questions in the same call.

### Large Forms (20 or More Gaps)

A large application (a state benefits portal, a multi-page application) can have 20
to 30 gaps. The count does not change the rules. It changes the presentation:

- Show the FULL gap-analysis table first, in one message. The user must see the
  full scope of the questions before the first question call. Do not show the
  gaps in small parts.
- Then ask with question calls in sequence. Group each call by a section of the
  form. Put the application name and the section name in each header (example:
  "SNAP — Income", "SNAP — Household").
- Thirty gaps take approximately eight question calls. That is correct. Do not
  remove a gap to make the list short.
- A conditional field depends on an answer (example: due date applies only when
  the person is pregnant). Ask the controlling question first. Ask the dependent
  question only when the answer makes the field applicable.

### Values You Must Not Derive

A fact next to a field is not the value of the field. Do not derive these values.
Ask for them:

- **A status from an address.** An address on file does not show that the person
  has stable housing. Many people without housing have a mailing address or a
  shelter address. Use only an explicit status value. When none exists, ask.
- **A contact preference from the available contacts.** A phone number on file
  does not show that the person prefers phone contact. When the form asks for a
  preferred contact method and no source states one, ask.
- **A household answer from a marital status.** "Single" does not show that the
  person lives alone.

The pattern is the same in each example: the form asks for a statement from the
person, and the record holds an adjacent fact. The fact does not answer the
question. These fields go in the MISSING group.

### How to Write the Questions

The user is a person who wants to complete a form. The user is not a developer.
Write each question at an easy reading level (approximately grade 6).

- Use the words that the form shows to the user. Use the person's name.
- Do not put element ids, selectors, attribute names, character limits, ref numbers,
  or browser terms in a question or in its options.
- Ask about one decision in each question.
- Give the effect of each option in plain words.
- Keep the technical detail in the transcript and in the playbook. It has value
  there, not in the question.

Bad question (too technical):

> The record shows a referral with verbal consent — that suggests someone is
> applying on the participant's behalf. Which path?
> 1. Maria is applying for herself — Sets chkBxSelfApplyYes. No representative
>    block.
> 2. On behalf — Sets chkBxSelfApplyNo, opening the representative block…

Good question (plain):

> Is Maria filling out this application herself, or is someone helping her apply?
> 1. She is applying herself.
> 2. Someone is helping her. (I will then ask for the helper's name and phone
>    number.)

Bad option (too technical):

> Append to first name — nameFirstTxt = "Maria LEE" (9 chars, fits the 25 limit)

Good option (plain):

> Put it with her first name, as "Maria LEE"

**WARNING: Do not fill a required field with an assumption. A form sent to a
government agency with invented data can cause legal and eligibility problems for a
real person. This rule has no exceptions. A "common baseline" or a "reasonable
default" is an assumption.**

## Part 2 — the Provenance Report (After the Fill)

After the readback in Phase 4, show the user a table of each value and its source.
Use these source categories:

| Source | Meaning | Example |
|---|---|---|
| USER | The user gave the value in this session. | The SSN from an answer to a question. |
| PAYLOAD | The value is in the source record, unchanged. | `"city": "<city>"` → `<city>`. |
| DERIVED | A stated rule changed a payload or user value. | The state name → the 2-letter code. `YYYY-MM-DD` → `MMDDYYYY`. |
| EMPTY | The field has no value. Give the reason. | SSN: the user gave no value. |

There is no ASSUMED category. An assumed value must not be in the form. If you find
an assumed value at report time, remove it from the form and mark the field EMPTY.

Report format:

The report goes to the user. Write the Field and Detail columns in plain words —
the words on the form, not selectors or attribute names.

```
| Field | Value in form | Source | Detail |
|---|---|---|---|
| First name | Maria | RECORD | from her record |
| State | CA | CHANGED | "California" written as "CA" — the form wants 2 letters |
| Social Security Number | (empty) | EMPTY | you did not give one; the form asks for it |
| Living arrangement | Own home | YOU | your answer |
```

The plain source words for the user: YOU (USER), RECORD (PAYLOAD), CHANGED
(DERIVED), EMPTY. Keep the technical category names in the transcript and in the
fill-agent reports; show the plain words to the user.

After the table, show these four items:

1. Payload values with no form field (the NO-FIELD group).
2. Required fields that are empty, with the reason for each.
3. New site facts that you added to the playbook in this session.
4. The approximate count of tool calls and question rounds in the run. The user
   tracks the cost of each run.

Show this report before the submit approval. The user must see the provenance of
each value before the user approves the submit step.
