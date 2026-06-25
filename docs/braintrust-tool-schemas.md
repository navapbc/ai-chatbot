# Braintrust Tool Schemas

Tool definitions for the web-automation agent, converted from the Vercel AI SDK `tool()` /
Zod definitions in `lib/ai/tools/` to JSON Schema for use in a Braintrust prompt.

These are the 11 tools wired into the agent in `app/(chat)/api/chat/route.ts` — names match the
keys the model actually sees.

## How to Use

In the Braintrust playground, open your prompt → **Add tools** → paste one tool object at a time.
Each block below is in the standard OpenAI function-tool shape
(`{ "type": "function", "function": { name, description, parameters } }`), which the playground
accepts. If the UI gives you separate **name** / **description** / **parameters** fields instead,
copy those three sub-values directly.

These are **schema-only** definitions — they let the model emit tool calls so you can test the
prompt's tool-selection and gap/summary behavior. They do **not** execute (no handler). To make a
tool actually run in Braintrust, define it with `project.tools.create({ ..., handler })` and push it
with the CLI — see `docs/BRAINTRUST_HOWTO.md`.

A combined array of all 11 tools is at the bottom for pasting in one shot if the UI supports it.

---

## getApricotRecord

```json
{
  "type": "function",
  "function": {
    "name": "getApricotRecord",
    "description": "Get a participant/client record from Apricot360 by record ID. Use this to fetch participant data for form filling.",
    "parameters": {
      "type": "object",
      "properties": {
        "recordId": {
          "type": "number",
          "description": "The unique record ID of the participant"
        }
      },
      "required": ["recordId"]
    }
  }
}
```

## getApricotForms

```json
{
  "type": "function",
  "function": {
    "name": "getApricotForms",
    "description": "Fetch forms from Apricot360 with optional pagination and filtering.",
    "parameters": {
      "type": "object",
      "properties": {
        "pageSize": {
          "type": "number",
          "description": "Number of forms to return per page (default: 25)"
        },
        "pageNumber": {
          "type": "number",
          "description": "Page number to retrieve (default: 1)"
        },
        "sort": {
          "type": "string",
          "description": "Field to sort by (e.g., \"name\", \"-name\" for descending)"
        },
        "filters": {
          "type": "object",
          "additionalProperties": { "type": "string" },
          "description": "Filters to apply"
        }
      }
    }
  }
}
```

## getApricotForm

```json
{
  "type": "function",
  "function": {
    "name": "getApricotForm",
    "description": "Get a specific form from Apricot360 by form ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "formId": {
          "type": "number",
          "description": "The unique ID of the form in Apricot360"
        }
      },
      "required": ["formId"]
    }
  }
}
```

## getApricotFormFields

```json
{
  "type": "function",
  "function": {
    "name": "getApricotFormFields",
    "description": "Get all fields for a specific form from Apricot360. Returns field definitions including labels, types, options, and validation requirements.",
    "parameters": {
      "type": "object",
      "properties": {
        "formId": {
          "type": "number",
          "description": "The unique ID of the form in Apricot360"
        }
      },
      "required": ["formId"]
    }
  }
}
```

## testApricotAuth

```json
{
  "type": "function",
  "function": {
    "name": "testApricotAuth",
    "description": "Test authentication with Apricot360 API. Use this to verify API credentials are working.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  }
}
```

## gapAnalysis

```json
{
  "type": "function",
  "function": {
    "name": "gapAnalysis",
    "description": "Shows the caseworker a card listing ONLY the missing fields, in the order they appear on the original form. Calling this tool ends your turn — do not call any other tools after it; wait for the caseworker's reply. Include only missing fields, no fields you already have. After calling, write one short sentence like \"Please provide the missing info above.\" and stop. If nothing is missing, do not call this tool.",
    "parameters": {
      "type": "object",
      "properties": {
        "formName": {
          "type": "string",
          "description": "Name of the form being filled, e.g. \"WIC Application\""
        },
        "clientName": {
          "type": "string",
          "description": "Full name of the participant the form is being filled for"
        },
        "missingFields": {
          "type": "array",
          "description": "Missing fields in the order they appear on the original form.",
          "items": {
            "type": "object",
            "properties": {
              "field": { "type": "string", "description": "Field label" },
              "options": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Possible answer options, if applicable"
              },
              "inputType": {
                "type": "string",
                "enum": ["text", "select", "date", "boolean", "textarea"],
                "description": "Expected input type"
              },
              "multiSelect": {
                "type": "boolean",
                "description": "Whether multiple options can be selected"
              },
              "condition": {
                "type": "string",
                "description": "Condition under which this field is required, e.g. \"if pregnant\""
              },
              "required": {
                "type": "boolean",
                "description": "Whether this field is required to submit the form"
              },
              "placeholder": {
                "type": "string",
                "description": "Placeholder hint shown inside the input"
              },
              "note": {
                "type": "string",
                "description": "Short helper text shown under the field label"
              }
            },
            "required": ["field"]
          }
        }
      },
      "required": ["missingFields"]
    }
  }
}
```

## formSummary

```json
{
  "type": "function",
  "function": {
    "name": "formSummary",
    "description": "Display a form summary card showing what was filled in and where each value came from. Call this INSTEAD of writing a summary message at the end of form completion. List fields in the order they appear on the original form. NEVER include CAPTCHA, reCAPTCHA, Turnstile, \"I'm not a robot\", or any bot-challenge widget — they are not form fields. Also exclude submit buttons, hidden inputs, and decorative text. The card already displays all information — do NOT write any text listing the fields before or after calling this tool. Just call the tool, then follow with one short sentence like \"Please review and submit when ready.\"",
    "parameters": {
      "type": "object",
      "properties": {
        "formName": {
          "type": "string",
          "description": "Name of the form that was filled, e.g. \"WIC Application\""
        },
        "clientName": {
          "type": "string",
          "description": "Full name of the participant the form was filled for"
        },
        "fields": {
          "type": "array",
          "description": "All form fields in the order they appear on the original form. Each field has a source indicating where the value came from.",
          "items": {
            "type": "object",
            "properties": {
              "field": { "type": "string", "description": "Field label" },
              "value": {
                "type": "string",
                "description": "Value that was filled in. Omit or leave empty for fields that could not be filled."
              },
              "source": {
                "type": "string",
                "enum": ["database", "caseworker", "inferred", "missing"],
                "description": "\"database\" = pulled from Apricot records, \"caseworker\" = provided by the caseworker this session, \"inferred\" = agent reasoned from available data, \"missing\" = field could not be filled"
              },
              "inputType": {
                "type": "string",
                "enum": ["text", "select", "radio", "checkbox"],
                "description": "Type of input the form field uses. Use \"select\" for dropdowns, \"radio\" for single-choice radio buttons, \"checkbox\" ONLY for fields that allow multiple simultaneous selections, or omit for plain text."
              },
              "options": {
                "type": "array",
                "items": { "type": "string" },
                "description": "REQUIRED for select/radio/checkbox fields. Every available choice exactly as the form labels it (e.g. [\"Yes\", \"No\"], [\"Male\", \"Female\", \"Non-binary\"]). The `value` you pass MUST match one of these strings character-for-character or the dropdown will render empty. Re-snapshot the form to read the real options — never guess."
              },
              "required": {
                "type": "boolean",
                "description": "Whether the field is required to submit the form"
              },
              "inferredFrom": {
                "type": "string",
                "description": "For inferred fields only: a short description of what the value was based on, e.g. \"the zipcode\", \"the client's date of birth\", \"the household size\""
              }
            },
            "required": ["field", "source"]
          }
        }
      },
      "required": ["fields"]
    }
  }
}
```

## actionLabel

```json
{
  "type": "function",
  "function": {
    "name": "actionLabel",
    "description": "Label the upcoming group of browser actions with a human-readable title. Call this ONCE before starting a sequence of related browser actions so the UI can show a meaningful group heading. Do NOT call it before every individual action — only at the start of a logical group. Examples: \"Filling in personal information\", \"Navigating to WIC portal\", \"Selecting household members\", \"Reviewing application before submission\".",
    "parameters": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "enum": ["fill", "navigate", "interact", "read", "search", "misc"],
          "description": "Type of action group, used to select the UI icon and label."
        }
      },
      "required": ["category"]
    }
  }
}
```

## browser

```json
{
  "type": "function",
  "function": {
    "name": "browser",
    "description": "Execute browser automation commands on a remote Kernel browser. Send structured JSON commands with an \"action\" field and action-specific parameters. See the Browser Automation skill for snapshot discipline, selector strategy, and workflow rules.\n\nCommands:\n- { action: \"navigate\", url: \"<url>\" } - Navigate to URL\n- { action: \"snapshot\" } - Full accessibility tree (ALWAYS do this first)\n- { action: \"snapshot\", selector: \"form\" } - Scoped snapshot (reduces noise)\n- { action: \"snapshot\", interactive: true } - Interactive elements only with refs\n- { action: \"click\", selector: \"@e1\" } - Click element by ref\n- { action: \"fill\", selector: \"@e1\", value: \"text\" } - Clear field and fill (programmatic — use for plain text fields)\n- { action: \"type\", selector: \"@e1\", text: \"text\", clear: true } - Simulate real keystrokes (use for masked fields: SSN, date, phone, state, zip)\n- { action: \"select\", selector: \"@e1\", values: [\"option\"] } - Select native dropdown option\n- { action: \"getbylabel\", label: \"Field Name\", subaction: \"fill\", value: \"val\" } - Fill by accessible label\n- { action: \"press\", key: \"Enter\" } - Press key (Tab, Escape, ArrowDown, etc.)\n- { action: \"hover\", selector: \"@e1\" } - Hover over element\n- { action: \"check\", selector: \"@e1\" } - Toggle checkbox on\n- { action: \"uncheck\", selector: \"@e1\" } - Toggle checkbox off\n- { action: \"scrollintoview\", selector: \"@e1\" } - Scroll element into view\n- { action: \"wait\", selector: \"@e1\" } - Wait for element\n- { action: \"wait\", timeout: 2000 } - Wait milliseconds\n- { action: \"waitforloadstate\", state: \"networkidle\" } - Wait for network to settle\n- { action: \"gettext\", selector: \"@e1\" } - Get element text content\n- { action: \"inputvalue\", selector: \"@e1\" } - Get input field value\n- { action: \"url\" } - Get current URL\n- { action: \"title\" } - Get page title\n- { action: \"scroll\", direction: \"down\", amount: 500 } - Scroll down 500px\n- { action: \"screenshot\" } - Take screenshot\n- { action: \"back\" } / { action: \"forward\" } - Browser navigation (AVOID during form filling — may wipe state)\n- { action: \"evaluate\", script: \"document.title\" } - Run JavaScript (ONLY for reading simple values — NEVER to find/click elements)\n- { action: \"tab_list\" } / { action: \"tab_switch\", index: N } / { action: \"tab_new\" } / { action: \"tab_close\" } - Tab management\n- { action: \"dialog\", response: \"accept\" } / { action: \"dialog\", response: \"dismiss\" } - Handle browser dialogs\n- { action: \"frame\", selector: \"#iframe\" } / { action: \"mainframe\" } - Switch between frames\n\nNEVER navigate away from the target application domain. Do NOT click social media links, share buttons, or external links.",
    "parameters": {
      "type": "object",
      "description": "Structured command object with action and action-specific parameters",
      "properties": {
        "action": { "type": "string", "description": "The command action (e.g. \"navigate\", \"click\", \"snapshot\", \"fill\")" },
        "selector": { "type": "string", "description": "Element selector: ref (@e1), CSS (#id), or label" },
        "value": { "type": "string", "description": "Value for fill action" },
        "text": { "type": "string", "description": "Text for type action" },
        "url": { "type": "string", "description": "URL for navigate action" },
        "key": { "type": "string", "description": "Key for press action (e.g. \"Enter\", \"Tab\")" },
        "label": { "type": "string", "description": "Label text for getbylabel action" },
        "subaction": { "type": "string", "description": "Sub-action for getbylabel (\"click\", \"fill\", \"check\")" },
        "script": { "type": "string", "description": "JavaScript for evaluate action" },
        "values": { "type": "array", "items": { "type": "string" }, "description": "Option values for select action — must be an array" },
        "timeout": { "type": "number", "description": "Timeout in ms for wait action — must be a number" },
        "amount": { "type": "number", "description": "Scroll amount in px — must be a number" },
        "delay": { "type": "number", "description": "Delay between keystrokes in ms — must be a number" },
        "interactive": { "type": "boolean", "description": "Show only interactive elements in snapshot — must be boolean" },
        "clear": { "type": "boolean", "description": "Clear field before typing — must be boolean" },
        "direction": { "type": "string", "description": "Scroll direction: \"up\" or \"down\"" },
        "state": { "type": "string", "description": "Load state for waitforloadstate (e.g. \"networkidle\")" },
        "index": { "type": "number", "description": "Tab index for tab_switch/tab_close" },
        "response": { "type": "string", "description": "Dialog response: \"accept\" or \"dismiss\"" },
        "promptText": { "type": "string", "description": "Text to enter in prompt dialog" }
      },
      "required": ["action"]
    }
  }
}
```

## checkSubmitGate

```json
{
  "type": "function",
  "function": {
    "name": "checkSubmitGate",
    "description": "Call when the submit button is disabled and the page has a Cloudflare Turnstile widget. Probes the DOM, then force-enables the button so the caseworker can take control and submit. Never clicks submit.",
    "parameters": {
      "type": "object",
      "properties": {
        "forceEnable": {
          "type": "boolean",
          "default": true,
          "description": "If true, after probing also force-enable the submit button (invoke Turnstile callback if present, then remove disabled attribute)."
        }
      }
    }
  }
}
```

## readReference

```json
{
  "type": "function",
  "function": {
    "name": "readReference",
    "description": "Load a reference document. Use the path the system prompt instructs you to load (e.g. \"field-patterns.md\", \"custom-dropdowns.md\", \"browser-commands.md\").",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Filename within lib/ai/prompts/references (e.g. \"field-patterns.md\")"
        }
      },
      "required": ["path"]
    }
  }
}
```

---

## All Tools (Combined Array)

```json
[
  {
    "type": "function",
    "function": {
      "name": "getApricotRecord",
      "description": "Get a participant/client record from Apricot360 by record ID. Use this to fetch participant data for form filling.",
      "parameters": {
        "type": "object",
        "properties": {
          "recordId": { "type": "number", "description": "The unique record ID of the participant" }
        },
        "required": ["recordId"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "getApricotForms",
      "description": "Fetch forms from Apricot360 with optional pagination and filtering.",
      "parameters": {
        "type": "object",
        "properties": {
          "pageSize": { "type": "number", "description": "Number of forms to return per page (default: 25)" },
          "pageNumber": { "type": "number", "description": "Page number to retrieve (default: 1)" },
          "sort": { "type": "string", "description": "Field to sort by (e.g., \"name\", \"-name\" for descending)" },
          "filters": { "type": "object", "additionalProperties": { "type": "string" }, "description": "Filters to apply" }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "getApricotForm",
      "description": "Get a specific form from Apricot360 by form ID.",
      "parameters": {
        "type": "object",
        "properties": {
          "formId": { "type": "number", "description": "The unique ID of the form in Apricot360" }
        },
        "required": ["formId"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "getApricotFormFields",
      "description": "Get all fields for a specific form from Apricot360. Returns field definitions including labels, types, options, and validation requirements.",
      "parameters": {
        "type": "object",
        "properties": {
          "formId": { "type": "number", "description": "The unique ID of the form in Apricot360" }
        },
        "required": ["formId"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "testApricotAuth",
      "description": "Test authentication with Apricot360 API. Use this to verify API credentials are working.",
      "parameters": { "type": "object", "properties": {} }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "gapAnalysis",
      "description": "Shows the caseworker a card listing ONLY the missing fields, in the order they appear on the original form. Calling this tool ends your turn — do not call any other tools after it; wait for the caseworker's reply. Include only missing fields, no fields you already have. After calling, write one short sentence like \"Please provide the missing info above.\" and stop. If nothing is missing, do not call this tool.",
      "parameters": {
        "type": "object",
        "properties": {
          "formName": { "type": "string", "description": "Name of the form being filled, e.g. \"WIC Application\"" },
          "clientName": { "type": "string", "description": "Full name of the participant the form is being filled for" },
          "missingFields": {
            "type": "array",
            "description": "Missing fields in the order they appear on the original form.",
            "items": {
              "type": "object",
              "properties": {
                "field": { "type": "string", "description": "Field label" },
                "options": { "type": "array", "items": { "type": "string" }, "description": "Possible answer options, if applicable" },
                "inputType": { "type": "string", "enum": ["text", "select", "date", "boolean", "textarea"], "description": "Expected input type" },
                "multiSelect": { "type": "boolean", "description": "Whether multiple options can be selected" },
                "condition": { "type": "string", "description": "Condition under which this field is required, e.g. \"if pregnant\"" },
                "required": { "type": "boolean", "description": "Whether this field is required to submit the form" },
                "placeholder": { "type": "string", "description": "Placeholder hint shown inside the input" },
                "note": { "type": "string", "description": "Short helper text shown under the field label" }
              },
              "required": ["field"]
            }
          }
        },
        "required": ["missingFields"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "formSummary",
      "description": "Display a form summary card showing what was filled in and where each value came from. Call this INSTEAD of writing a summary message at the end of form completion. List fields in the order they appear on the original form. NEVER include CAPTCHA, reCAPTCHA, Turnstile, \"I'm not a robot\", or any bot-challenge widget — they are not form fields. Also exclude submit buttons, hidden inputs, and decorative text. The card already displays all information — do NOT write any text listing the fields before or after calling this tool. Just call the tool, then follow with one short sentence like \"Please review and submit when ready.\"",
      "parameters": {
        "type": "object",
        "properties": {
          "formName": { "type": "string", "description": "Name of the form that was filled, e.g. \"WIC Application\"" },
          "clientName": { "type": "string", "description": "Full name of the participant the form was filled for" },
          "fields": {
            "type": "array",
            "description": "All form fields in the order they appear on the original form. Each field has a source indicating where the value came from.",
            "items": {
              "type": "object",
              "properties": {
                "field": { "type": "string", "description": "Field label" },
                "value": { "type": "string", "description": "Value that was filled in. Omit or leave empty for fields that could not be filled." },
                "source": { "type": "string", "enum": ["database", "caseworker", "inferred", "missing"], "description": "\"database\" = pulled from Apricot records, \"caseworker\" = provided by the caseworker this session, \"inferred\" = agent reasoned from available data, \"missing\" = field could not be filled" },
                "inputType": { "type": "string", "enum": ["text", "select", "radio", "checkbox"], "description": "Type of input the form field uses. Use \"select\" for dropdowns, \"radio\" for single-choice radio buttons, \"checkbox\" ONLY for fields that allow multiple simultaneous selections, or omit for plain text." },
                "options": { "type": "array", "items": { "type": "string" }, "description": "REQUIRED for select/radio/checkbox fields. Every available choice exactly as the form labels it. The value you pass MUST match one of these strings character-for-character or the dropdown will render empty." },
                "required": { "type": "boolean", "description": "Whether the field is required to submit the form" },
                "inferredFrom": { "type": "string", "description": "For inferred fields only: a short description of what the value was based on, e.g. \"the zipcode\", \"the client's date of birth\", \"the household size\"" }
              },
              "required": ["field", "source"]
            }
          }
        },
        "required": ["fields"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "actionLabel",
      "description": "Label the upcoming group of browser actions with a human-readable title. Call this ONCE before starting a sequence of related browser actions so the UI can show a meaningful group heading. Do NOT call it before every individual action — only at the start of a logical group.",
      "parameters": {
        "type": "object",
        "properties": {
          "category": { "type": "string", "enum": ["fill", "navigate", "interact", "read", "search", "misc"], "description": "Type of action group, used to select the UI icon and label." }
        },
        "required": ["category"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "browser",
      "description": "Execute browser automation commands on a remote Kernel browser. Send structured JSON commands with an \"action\" field and action-specific parameters. Actions include navigate, snapshot, click, fill, type, select, getbylabel, press, hover, check, uncheck, scrollintoview, wait, waitforloadstate, gettext, inputvalue, url, title, scroll, screenshot, back, forward, evaluate, tab management, dialog, frame. ALWAYS snapshot first. NEVER navigate away from the target application domain.",
      "parameters": {
        "type": "object",
        "description": "Structured command object with action and action-specific parameters",
        "properties": {
          "action": { "type": "string", "description": "The command action (e.g. \"navigate\", \"click\", \"snapshot\", \"fill\")" },
          "selector": { "type": "string", "description": "Element selector: ref (@e1), CSS (#id), or label" },
          "value": { "type": "string", "description": "Value for fill action" },
          "text": { "type": "string", "description": "Text for type action" },
          "url": { "type": "string", "description": "URL for navigate action" },
          "key": { "type": "string", "description": "Key for press action (e.g. \"Enter\", \"Tab\")" },
          "label": { "type": "string", "description": "Label text for getbylabel action" },
          "subaction": { "type": "string", "description": "Sub-action for getbylabel (\"click\", \"fill\", \"check\")" },
          "script": { "type": "string", "description": "JavaScript for evaluate action" },
          "values": { "type": "array", "items": { "type": "string" }, "description": "Option values for select action — must be an array" },
          "timeout": { "type": "number", "description": "Timeout in ms for wait action — must be a number" },
          "amount": { "type": "number", "description": "Scroll amount in px — must be a number" },
          "delay": { "type": "number", "description": "Delay between keystrokes in ms — must be a number" },
          "interactive": { "type": "boolean", "description": "Show only interactive elements in snapshot — must be boolean" },
          "clear": { "type": "boolean", "description": "Clear field before typing — must be boolean" },
          "direction": { "type": "string", "description": "Scroll direction: \"up\" or \"down\"" },
          "state": { "type": "string", "description": "Load state for waitforloadstate (e.g. \"networkidle\")" },
          "index": { "type": "number", "description": "Tab index for tab_switch/tab_close" },
          "response": { "type": "string", "description": "Dialog response: \"accept\" or \"dismiss\"" },
          "promptText": { "type": "string", "description": "Text to enter in prompt dialog" }
        },
        "required": ["action"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "checkSubmitGate",
      "description": "Call when the submit button is disabled and the page has a Cloudflare Turnstile widget. Probes the DOM, then force-enables the button so the caseworker can take control and submit. Never clicks submit.",
      "parameters": {
        "type": "object",
        "properties": {
          "forceEnable": { "type": "boolean", "default": true, "description": "If true, after probing also force-enable the submit button (invoke Turnstile callback if present, then remove disabled attribute)." }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "readReference",
      "description": "Load a reference document. Use the path the system prompt instructs you to load (e.g. \"field-patterns.md\", \"custom-dropdowns.md\", \"browser-commands.md\").",
      "parameters": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "Filename within lib/ai/prompts/references (e.g. \"field-patterns.md\")" }
        },
        "required": ["path"]
      }
    }
  }
]
```
