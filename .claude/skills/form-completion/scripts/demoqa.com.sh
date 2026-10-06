#!/usr/bin/env bash
# demoqa.com.sh — the confirmed fill sequence for https://demoqa.com/automation-practice-form
#
# WORKED EXAMPLE of a site script. A playbook tells an agent what to do in prose; a site
# script does it. The difference matters more than it looks: prose has to be interpreted,
# and interpretation is where a fill agent gets lost — hunting for a widget, re-opening the
# page, spending a hundred commands on a form it already had documented. A script turns the
# whole sequence into commands nobody has to reason about, which is also what makes a
# small, cheap model safe to run it.
#
# This site is a public practice form with no real backend. It is here as the reference
# shape for a real one, not because anyone needs to fill it.
#
# Usage:
#   SESSION=demoqa source .claude/skills/form-completion/scripts/demoqa.com.sh
#   demoqa_fill "Maria" "Garcia" "maria@example.org" Female 5551234567 \
#               "15 Mar 1990" Maths Reading "742 Evergreen Terrace" NCR Delhi
#   demoqa_readback          # prints field=value lines; compare against what you sent
#
# Every function returns non-zero on a failure it can detect. A non-zero return is a
# BLOCKED condition: report it, do not explore for a workaround, and never re-open the page.
set -uo pipefail

: "${SESSION:?set SESSION to the agent-browser session name}"
: "${AGENT_BROWSER_BIN:=agent-browser}"

_ab() { "$AGENT_BROWSER_BIN" --session "$SESSION" "$@"; }

DEMOQA_URL="https://demoqa.com/automation-practice-form"

# --- freshness probe -------------------------------------------------------
# Three ids that must exist. If any is missing the site changed: the playbook is stale,
# this script is stale with it, and the run needs a cold start rather than a repair.
demoqa_probe() {
  local id
  for id in firstName userEmail currentAddress; do
    [ "$(_ab get count "#$id" 2>/dev/null)" = "1" ] || { echo "STALE: #$id not found" >&2; return 1; }
  done
}

# --- date of birth ---------------------------------------------------------
# Never fill or type this field. It is pre-filled with today's date and is a
# react-datepicker trigger; typing over it concatenates old and new text and the tool
# reports success. Drive the widget instead.
demoqa_dob() {
  local month="$1" year="$2" day="$3"        # e.g. March 1990 15
  _ab click "#dateOfBirthInput" || return 1
  _ab select ".react-datepicker__month-select" "$month" || return 1
  _ab select ".react-datepicker__year-select" "$year"  || return 1
  # The day cell's aria-label carries a weekday name we must not guess. Read it back from
  # the rendered popup and match on the day number, then click that exact label.
  local pad label
  pad="$(printf '%03d' "$day")"
  label="$(_ab snapshot -i 2>/dev/null \
           | grep -o "\"Choose [^\"]*${month} ${day}[a-z][a-z], ${year}\"" \
           | head -1 | tr -d '"')"
  if [ -z "$label" ]; then
    echo "BLOCKED: no day cell for ${month} ${day}, ${year} (looked for aria-label)" >&2
    return 1
  fi
  _ab click "[aria-label=\"${label}\"]" || return 1
  : "$pad"
}

# --- react-select (State, City) --------------------------------------------
# There are no native <select> elements on this form. Click the container, filter with the
# inner text input, commit the auto-focused option with Enter.
demoqa_rselect() {
  local container="$1" input="$2" value="$3"
  _ab click "$container" || return 1
  _ab type "$input" "$value" || return 1
  _ab key Enter || return 1
}

# --- subjects typeahead ----------------------------------------------------
demoqa_subject() {
  local value="$1"
  _ab click "#subjectsInput" || return 1
  _ab type "#subjectsInput" "$value" || return 1
  _ab key Enter || return 1
}

# --- the whole form --------------------------------------------------------
demoqa_fill() {
  local first="$1" last="$2" email="$3" gender="$4" mobile="$5" dob="$6" \
        subject="$7" hobby="$8" address="$9" state="${10}" city="${11}"

  demoqa_probe || return 1

  local grad hrad
  case "$gender" in
    Male)   grad="#gender-radio-1" ;;
    Female) grad="#gender-radio-2" ;;
    Other)  grad="#gender-radio-3" ;;
    *) echo "BLOCKED: unknown gender '$gender'" >&2; return 1 ;;
  esac
  case "$hobby" in
    Sports)  hrad="#hobbies-checkbox-1" ;;
    Reading) hrad="#hobbies-checkbox-2" ;;
    Music)   hrad="#hobbies-checkbox-3" ;;
    *) echo "BLOCKED: unknown hobby '$hobby'" >&2; return 1 ;;
  esac

  # Plain fields take a straight fill and read back immediately — no mask, no gate.
  _ab fill "#firstName"      "$first"   || return 1
  _ab fill "#lastName"       "$last"    || return 1
  _ab fill "#userEmail"      "$email"   || return 1
  _ab fill "#userNumber"     "$mobile"  || return 1
  _ab fill "#currentAddress" "$address" || return 1
  _ab check "$grad" || return 1
  _ab check "$hrad" || return 1

  # "15 Mar 1990" -> day=15 monthname=March year=1990
  local d m y monthname
  read -r d m y <<< "$dob"
  case "$m" in
    Jan*) monthname=January ;; Feb*) monthname=February ;; Mar*) monthname=March ;;
    Apr*) monthname=April ;;   May*) monthname=May ;;      Jun*) monthname=June ;;
    Jul*) monthname=July ;;    Aug*) monthname=August ;;   Sep*) monthname=September ;;
    Oct*) monthname=October ;; Nov*) monthname=November ;; Dec*) monthname=December ;;
    *) echo "BLOCKED: cannot parse month from '$dob'" >&2; return 1 ;;
  esac
  demoqa_dob "$monthname" "$y" "$((10#$d))" || return 1

  demoqa_subject "$subject" || return 1
  demoqa_rselect "#state" "#react-select-3-input" "$state" || return 1
  demoqa_rselect "#city"  "#react-select-4-input" "$city"  || return 1
}

# --- readback --------------------------------------------------------------
# One pass, every field, printed as field=value. Each field type needs a different read:
# plain fields have a value, radios and checkboxes have a checked state, and the two
# react-select widgets and the subjects tag only exist as rendered text.
demoqa_readback() {
  local id
  for id in firstName lastName userEmail userNumber currentAddress dateOfBirthInput; do
    printf '%s=%s\n' "$id" "$(_ab get value "#$id" 2>/dev/null)"
  done
  for id in gender-radio-1 gender-radio-2 gender-radio-3 \
            hobbies-checkbox-1 hobbies-checkbox-2 hobbies-checkbox-3; do
    printf '%s=%s\n' "$id" "$(_ab is checked "#$id" 2>/dev/null)"
  done
  printf 'subjects=%s\n' "$(_ab get text "#subjectsContainer" 2>/dev/null | tr -s '[:space:]' ' ')"
  printf 'state=%s\n'    "$(_ab get text "#state" 2>/dev/null | tr -s '[:space:]' ' ')"
  printf 'city=%s\n'     "$(_ab get text "#city"  2>/dev/null | tr -s '[:space:]' ' ')"
}

# Submit is deliberately absent from this file. Submission is Phase 6 and belongs to the
# orchestrator with a person approving it, never to a script.
