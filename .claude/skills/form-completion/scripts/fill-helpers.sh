#!/usr/bin/env bash
# Batch helpers for form fills with agent-browser.
# Source this file. Set SESSION. Put all the write calls in one Bash invocation.
# Then do one readback with V. Do not skip the readback.
#
#   source .claude/skills/form-completion/scripts/fill-helpers.sh
#   SESSION=localchrome
#   S fill "#firstName" "Maria"
#   K "#birthDate" "MMDDYYYY"           # Masked field: one key for each character
#   C "#agreeYes"                       # Idempotent check
#   V firstName birthDate               # Readback. Always do this.
#
# Cold-start survey (read-only, no writes):
#   SURVEY                              # Count of input/select/textarea/iframe/form
#   IFRAMES                             # The src of each iframe
#   FIELDS "body"                       # Full field map: type, id, value, label
#   ATTRS "required maxlength" id1 id2   # Attribute probe. Prints "-" when absent.
#   OPTIONS "#someSelect"               # All the option texts

AB="${AGENT_BROWSER_BIN:-./node_modules/.bin/agent-browser}"
SESSION="${SESSION:-form-fill}"

# One command, with the output removed
S() { "$AB" --session "$SESSION" "$@" >/dev/null 2>&1; }

# Fill a masked field with one keypress for each character: K "#selector" "chars"
# Use fold, not `read -n1`. The `read -n1` option fails in zsh.
# The first three keys clear the field with real keystrokes. The `fill ""` command
# can put the mask buffer in a bad state. Then the field ignores keys, or the caret
# stops at the end. The keyboard clear corrects this. It causes no damage on a clean
# field. Always keep these three keys.
case "$(uname)" in Darwin) _SELALL="Meta+a";; *) _SELALL="Control+a";; esac
K() {
  local sel="$1" chars="$2" c
  S click "$sel"
  S key "$_SELALL"; S key "Backspace"; S key "Home"
  for c in $(printf '%s' "$chars" | fold -w1); do S key "$c"; done
}

# Set a checkbox (idempotent)
C() { S check "$1"; }
U() { S uncheck "$1"; }

# Readback of field values by id, without the leading "#": V id1 id2 id3 ...
V() {
  local f
  for f in "$@"; do
    printf '%s = ' "$f"
    "$AB" --session "$SESSION" get value "#$f" 2>&1 | tail -1
  done
}

# Page survey in ONE tool call: SURVEY
# Prints the count of each element class. Use this first on a cold start.
# A high iframe count does NOT mean that the form is in an iframe. Use IFRAMES next.
SURVEY() {
  local s
  for s in input select textarea iframe form button; do
    printf '%-10s = ' "$s"
    "$AB" --session "$SESSION" get count "$s" 2>&1 | tail -1
  done
}

# List the src of each iframe: IFRAMES
# Do NOT use iframe:nth-of-type(N) for this. The pseudo-class counts inside one
# parent, so N greater than 1 gives "Element not found" when the iframes have
# different parents. Parse the page HTML instead.
# reCAPTCHA and ad-tracker srcs hold no application field. Skip them.
IFRAMES() {
  "$AB" --session "$SESSION" get html "body" 2>/dev/null |
    grep -o '<iframe[^>]*src="[^"]*"' | sed 's/.*src="//;s/"$//' | cut -c1-120
}

# Attribute probe for a list of ids: ATTRS "attr1 attr2" id1 id2 id3 ...
# CAUTION: agent-browser prints "✓ Done" and no value when the attribute is ABSENT.
# This helper prints "-" in that case, so an absent attribute cannot look like a value.
ATTRS() {
  local attrs="$1"; shift
  local f a v
  for f in "$@"; do
    printf '%-50s' "$f"
    for a in $attrs; do
      v=$("$AB" --session "$SESSION" get attr "#$f" "$a" 2>&1 | tail -1)
      case "$v" in *"Done"*|"") v="-";; esac
      printf ' %s=%-6s' "$a" "$v"
    done
    printf '\n'
  done
}

# List all the options of a select: OPTIONS "#selector"
# Do not use `get text`: it gives one text block. Parse the HTML.
# CAUTION: an option text can occur two times in one select. Match by text, not by
# index.
OPTIONS() {
  "$AB" --session "$SESSION" get html "$1" 2>/dev/null |
    grep -o '<option[^>]*>[^<]*' | sed 's/<option[^>]*>//'
}

# Field inventory in ONE tool call: FIELDS [container-css]
# Prints one line for each input/select/textarea: type, id, name, value, label text.
# Use this for the cold-start field map and for groups that share one id
# (example: checkboxes with id=chkBxHealthHistory and different value attributes).
# Do not scan with get count in a loop. Do not use eval.
FIELDS() {
  local sel="${1:-form}"
  "$AB" --session "$SESSION" get html "$sel" 2>/dev/null | python3 -c '
import sys, html.parser

class P(html.parser.HTMLParser):
    # Checkbox and radio labels FOLLOW the input. Text and select labels COME BEFORE.
    def __init__(self):
        super().__init__(); self.out=[]; self.pending=[]; self.intag=None; self.last=""
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag in ("input","select","textarea"):
            t=a.get("type",tag)
            row={"tag":tag,"type":t,"id":a.get("id",""),
                 "name":a.get("name",""),"value":a.get("value",""),"label":""}
            if t in ("checkbox","radio"):
                self.pending.append(row)      # label is the next text node
            else:
                row["label"]=self.last        # label is the previous text node
            self.out.append(row)
        if tag=="option" and self.out and self.out[-1]["tag"]=="select":
            self.intag="option"
    def handle_data(self, data):
        t=" ".join(data.split())
        if not t: return
        if self.intag=="option":
            self.out[-1]["label"]=(self.out[-1]["label"]+" / "+t).strip(" /")[:200]
            return
        for row in self.pending:
            if not row["label"]: row["label"]=t[:80]
        self.pending=[]; self.last=t[:80]
    def handle_endtag(self, tag):
        if tag=="option": self.intag=None

p=P(); p.feed(sys.stdin.read())
for r in p.out:
    if r["id"].startswith("goog-gt") or r["name"] in ("sl","tl","query","gtrans","vote"):
        continue  # Google Translate widget noise
    t=r["type"]; i=r["id"] or "-"; v=r["value"] or "-"; l=r["label"] or "-"
    print("%-12s id=%-28s value=%-6s label=%s" % (t,i,v,l))
'
}
