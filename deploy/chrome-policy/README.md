# Chrome Policy For The Browser Bridge Extension

Templates and notes for installing the Labs ASP Browser Bridge extension on
managed Chrome by policy (force-install from a self-hosted update URL). IT
applies the policy through its own tooling (Google Admin console, Group
Policy/ADMX, or an MDM profile); nothing in this repository applies it.

Spec references below are to `docs/specs/2026-10-23-browser-extension-design.md`
("Managed Chrome Requirements", lines 398-443, and "Open Items for the Owner",
lines 481-499).

## Files

- `extension-policy.template.json`: the two policies, with the placeholders
  `__EXTENSION_ID__` and `__UPDATE_URL__`.
- `scripts/package-extension.ts` (repo root): builds the release package that
  the update URL serves. See "Building The Package" below.

## The Policies

### ExtensionInstallForcelist

```json
"ExtensionInstallForcelist": ["__EXTENSION_ID__;__UPDATE_URL__"]
```

Each entry is `<extension id>;<update manifest URL>`. Chrome installs the
extension from the update manifest, keeps it installed and the user cannot
remove or disable it. Off-store force-install is honored only on managed
machines (on Windows, domain-joined or MDM-managed; on macOS, MDM-managed);
confirm with IT for the pilot machines.

### ExtensionSettings

```json
"ExtensionSettings": {
  "__EXTENSION_ID__": {
    "installation_mode": "force_installed",
    "update_url": "__UPDATE_URL__"
  }
}
```

The same install expressed in the newer per-extension policy. Either policy is
enough on its own; the template sets both so it works with tooling that only
offers one of them. If IT already manages `ExtensionSettings`, merge this entry
into the existing object instead of replacing it.

What the template leaves out on purpose:

- **No `runtime_blocked_hosts`** for this extension, and none under `"*"` that
  would apply to it. From Chrome 155, any non-empty `runtime_blocked_hosts` for
  the extension makes `chrome.debugger.attach()` fail on every target (see
  "Chrome 155 Restrictions" below).
- **No `blocked_permissions`.** The extension needs `debugger`, `tabs`,
  `storage`, `sidePanel` and `alarms` (`extension/manifest.json`); blocking any
  of them breaks it.
- No `runtime_allowed_hosts`: the extension does not need it, and it does not
  undo a `runtime_blocked_hosts` entry on Chrome 155 and later.

## Filling The Placeholders

- `__EXTENSION_ID__`: the contents of `extension/EXTENSION_ID`
  (`ppcenadbeddfnlgklnccjkgjegmbkkkl` today). It is derived from the manifest
  `key`; the package script checks the two agree. If a Chrome Web Store listing
  ever assigns another id, the manifest `key`, `extension/EXTENSION_ID`,
  `RELAY_EXTENSION_ID` and `RELAY_EXTENSION_ORIGIN` must all change together.
- `__UPDATE_URL__`: the https URL where `updates.xml` is hosted. It must be the
  same value passed to the package script as `--update-url`.

```sh
sed -e "s|__EXTENSION_ID__|$(cat extension/EXTENSION_ID)|g" \
    -e "s|__UPDATE_URL__|https://downloads.example.org/labs-asp/updates.xml|g" \
    deploy/chrome-policy/extension-policy.template.json > extension-policy.json
```

In a sed replacement `&` stands for the matched text, so this command is only
correct for an update URL with no `&` (escape it as `\&` otherwise).

## Building The Package

```sh
pnpm exec tsx scripts/package-extension.ts \
  --app-origin https://app.example.org \
  --out ./release \
  --update-url https://downloads.example.org/labs-asp/updates.xml \
  --package-url https://downloads.example.org/labs-asp/extension.crx \
  --version 0.1.1
```

- Runs the release build (`EXTENSION_RELEASE=1`) for one app origin. The
  extension is built for exactly one origin, so dev and production need
  separate packages.
- Writes `extension.zip` (the five release files only), and `updates.xml` when
  both URLs are given. With `--update-url` the packaged manifest also carries
  `update_url`, which Chrome uses for later updates. Leave both URLs off for a
  zip meant for a Chrome Web Store upload: the store rejects `update_url`.
- Writes `extension.crx` only when `EXTENSION_PEM_PATH` names the extension's
  private key in a file outside the repository and that key matches the
  manifest `key`. Self-hosted force-install needs the CRX; the zip alone is not
  installable from an update URL. The key is never in the repository, in CI or
  in the release workflow; the owner signs on a machine that holds it.
- The hosted CRX and `updates.xml` hold no secret: the app origin is the only
  value built in, and there are no tokens. They must be reachable over https by
  Chrome's updater on the caseworkers' machines (a private bucket behind
  sign-in is not).

`.github/workflows/extension-release.yml` runs the same script by hand
(`workflow_dispatch`) and uploads the zip (and `updates.xml` when both URLs
are given) as workflow artifacts. It publishes nothing and has no signing key,
so it never produces a CRX.

## Chrome 155 Restrictions

From the spec, lines 407-423:

- Starting with Chrome 155, "`chrome.debugger.attach()` is rejected on
  enterprise-managed browsers" when (lines 411-413) `ExtensionSettings` sets a
  "non-empty `runtime_blocked_hosts`" for the extension: "Attach fails on all
  targets, even if `runtime_allowed_hosts` lists origins. Error: 'Host access
  is restricted by policy.'"
- Or when (lines 414-416) "The `DisableScreenshots` policy is on, or a Data Loss
  Prevention rule disables screenshot capture. Error: 'Screenshot capture is
  restricted by policy.'"
- Lines 419-422: launching Chrome with
  `--disable-features=ExtensionDebuggerStrictPolicyRestrictions` restores the
  old behavior but "is reported to be removed in Chrome 160, so it is not a
  plan." Unmanaged profiles are unaffected.

IT must confirm none of these apply to the pilot users (spec Open Item 8,
lines 495-497). The template sets none of them; a policy applied elsewhere in
the organization can still set them.

## The Debugging Banner

Chrome shows "Labs ASP Browser Bridge started debugging this browser" while
the extension is attached to a tab. From the spec, lines 425-444:

- Line 429: "Reported by third parties (UiPath documentation and a user issue
  on a similar extension)": force-install with `ExtensionInstallForcelist`
  "does not show the banner", and launching Chrome with
  `--silent-debugger-extension-api` "silences it".
- Lines 433-435: Chrome's own `chrome.debugger` reference mentions
  `--silent-debugger-extension-api` only for attaching to extension background
  pages, "and does not describe it as hiding the banner for tabs."
- Lines 436-437: a 2017 Chromium review discussed a policy preference for
  policy-installed extensions; "Whether it shipped, and its name, is
  unconfirmed."
- Line 438: "No documented, supported policy that hides the banner for a named
  extension was found."
- Lines 441-444: the panel explains the banner on the Ready screen, so the
  design does not depend on suppression. The flag "needs IT to control how
  Chrome is started, which may not be possible for every caseworker's browser"
  (lines 443-444). Whether IT can set a launch flag is spec Open Item 9
  (lines 498-499).

`--silent-debugger-extension-api` is a command-line flag, not a policy. It has
to be added to how Chrome is launched (shortcut, launcher or managed start
script), which IT controls.

## DeveloperToolsAvailability

The spec does not mention `DeveloperToolsAvailability`, and this project has
not checked how any of its values interacts with `chrome.debugger` for a
force-installed extension. Ask IT which value the pilot machines set
(including whether it is left at its default) and test the extension under
that exact setting before the pilot.

## Untested: Check On A Managed Machine

None of the following has been tested in this project (spec lines 401-402:
"None of the banner suppression below has been tested"). Check each on a pilot
machine before telling caseworkers what to expect:

1. The extension force-installs from the self-hosted `updates.xml` and CRX, and
   updates when the version in `updates.xml` increases.
2. Whether the debugging banner appears for a force-installed extension.
3. Whether `--silent-debugger-extension-api` hides the banner for tab targets.
4. That `chrome.debugger.attach()` succeeds under the organization's actual
   policies on Chrome 155 or later (no `runtime_blocked_hosts`,
   `DisableScreenshots` or screenshot DLP rule reaches this extension).
5. Whether the `DeveloperToolsAvailability` value IT sets affects
   `chrome.debugger`.
6. The pilot machines run at least the manifest's `minimum_chrome_version`
   (`120`); older Chrome refuses to install the extension.
