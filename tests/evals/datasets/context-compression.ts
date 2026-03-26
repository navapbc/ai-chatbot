import type { ModelMessage } from 'ai';

/**
 * Each test case provides a realistic message history from a benefits
 * form-filling session. The `expectedPreserved` array lists facts that
 * MUST appear in the compression summary for the eval to pass.
 */
export interface CompressionTestCase {
  name: string;
  description: string;
  messages: ModelMessage[];
  /** Facts that must be preserved in any valid summary */
  expectedPreserved: string[];
  /** Facts that must NOT appear (e.g. raw HTML, browser snapshots) */
  expectedOmitted?: string[];
}

// ---------------------------------------------------------------------------
// Helpers to build realistic message sequences
// ---------------------------------------------------------------------------

function userMsg(text: string): ModelMessage {
  return { role: 'user', content: text };
}

function assistantMsg(text: string): ModelMessage {
  return { role: 'assistant', content: text };
}

function toolResultMsg(toolName: string, result: Record<string, unknown>): ModelMessage {
  return {
    role: 'tool',
    content: [{ type: 'tool-result', toolCallId: 'call-1', toolName, result } as any],
  };
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

export const compressionDataset: CompressionTestCase[] = [
  {
    name: 'basic-participant-data',
    description: 'Preserves all participant field-value pairs from Apricot lookup',
    messages: [
      userMsg('Start SNAP application for participant Maria Garcia'),
      assistantMsg('Looking up participant data in Apricot...'),
      toolResultMsg('apricot', {
        firstName: 'Maria',
        lastName: 'Garcia',
        dob: '1985-03-15',
        ssn: '***-**-1234',
        address: '123 Main St, Springfield, IL 62701',
        phone: '555-0142',
        householdSize: 3,
        monthlyIncome: 1850,
        employerName: 'Springfield Foods Inc',
      }),
      assistantMsg(
        'Found participant record. Maria Garcia, DOB 1985-03-15, SSN ending 1234. ' +
        'Address: 123 Main St, Springfield, IL 62701. Household size: 3, monthly income: $1,850. ' +
        'Employer: Springfield Foods Inc. Starting SNAP application now.'
      ),
      userMsg('Looks correct, go ahead'),
      assistantMsg('Navigating to the SNAP application form...'),
      toolResultMsg('browser', {
        snapshot: '<html>...very long HTML snapshot of the SNAP form...</html>',
        url: 'https://benefits.state.il.us/snap/apply',
      }),
      assistantMsg('I can see the SNAP application form. Filling in personal information section.'),
      assistantMsg('Filled first name: Maria, last name: Garcia, DOB: 03/15/1985'),
      userMsg('Her phone number actually changed to 555-0199'),
      assistantMsg('Got it, updating phone to 555-0199 instead of the Apricot value 555-0142.'),
      // Padding messages to exceed KEEP_RECENT threshold
      assistantMsg('Moving to the address section of the form.'),
      assistantMsg('Filled address: 123 Main St, Springfield, IL 62701'),
      userMsg('Continue to household section'),
      assistantMsg('Now filling household information. Household size: 3.'),
      assistantMsg('Entering monthly income: $1,850 from Springfield Foods Inc.'),
    ],
    expectedPreserved: [
      'Maria Garcia',
      '1985-03-15',
      '1234',          // SSN last 4
      '123 Main St',
      'Springfield',
      'household size',
      '1850',
      'Springfield Foods Inc',
      '555-0199',       // corrected phone
      'SNAP',
    ],
    expectedOmitted: [
      '<html>',          // raw HTML should be pruned
    ],
  },

  {
    name: 'gap-analysis-preservation',
    description: 'Preserves gap analysis results and caseworker responses',
    messages: [
      userMsg('Apply for Medicaid for John Smith'),
      assistantMsg('Loading participant data...'),
      toolResultMsg('apricot', {
        firstName: 'John',
        lastName: 'Smith',
        dob: '1990-07-22',
        ssn: '***-**-5678',
        address: '456 Oak Ave, Chicago, IL 60601',
        citizenshipStatus: 'US Citizen',
      }),
      assistantMsg('Found John Smith. DOB 1990-07-22, SSN ending 5678. US Citizen.'),
      assistantMsg('Running gap analysis on Medicaid application fields...'),
      toolResultMsg('gap-analysis', {
        gaps: [
          { field: 'monthlyIncome', reason: 'Not in Apricot record' },
          { field: 'employerName', reason: 'Not in Apricot record' },
          { field: 'disabilityStatus', reason: 'Required for Medicaid, not in record' },
        ],
      }),
      assistantMsg(
        'Gap analysis found 3 missing fields:\n' +
        '1. Monthly income — not in Apricot record\n' +
        '2. Employer name — not in Apricot record\n' +
        '3. Disability status — required for Medicaid, not in record\n' +
        'Can you provide these?'
      ),
      userMsg('Monthly income is $2,400 from TechCorp. No disability.'),
      assistantMsg(
        'Thank you. Recording:\n' +
        '- Monthly income: $2,400\n' +
        '- Employer: TechCorp\n' +
        '- Disability status: None\n' +
        'All gaps resolved. Continuing with form.'
      ),
      assistantMsg('Navigating to Medicaid application...'),
      toolResultMsg('browser', {
        snapshot: '<div id="medicaid-form">...</div>',
        url: 'https://benefits.state.il.us/medicaid/apply',
      }),
      assistantMsg('Filling personal info: John Smith, DOB 07/22/1990'),
      assistantMsg('Entering income: $2,400/month, employer: TechCorp'),
      assistantMsg('Disability status: None'),
      userMsg('Looks good, continue'),
      assistantMsg('Moving to the citizenship section. Entering: US Citizen.'),
    ],
    expectedPreserved: [
      'John Smith',
      '1990-07-22',
      '5678',
      'monthlyIncome',
      'employerName',
      'disabilityStatus',
      '2400',
      'TechCorp',
      'US Citizen',
      'Medicaid',
      'gap',
    ],
  },

  {
    name: 'caseworker-corrections',
    description: 'Preserves caseworker corrections that override database values',
    messages: [
      userMsg('Start TANF application for Rosa Hernandez'),
      assistantMsg('Looking up participant...'),
      toolResultMsg('apricot', {
        firstName: 'Rosa',
        lastName: 'Hernandez',
        dob: '1978-11-03',
        address: '789 Pine Rd, Peoria, IL 61602',
        householdSize: 5,
        monthlyIncome: 1200,
      }),
      assistantMsg('Found Rosa Hernandez. DOB 1978-11-03, address 789 Pine Rd Peoria IL. Household: 5, income $1,200.'),
      userMsg('The household size is wrong, it should be 4 now. One child moved out.'),
      assistantMsg('Noted — updating household size from 5 to 4 per caseworker correction.'),
      userMsg('Also her income went up to $1,500'),
      assistantMsg('Updated monthly income from $1,200 to $1,500 per caseworker.'),
      assistantMsg('Starting TANF application with corrected values...'),
      toolResultMsg('browser', {
        snapshot: '<form>long HTML...</form>',
        url: 'https://benefits.state.il.us/tanf/apply',
      }),
      assistantMsg('Filling form with: Rosa Hernandez, DOB 11/03/1978, household size 4, income $1,500'),
      assistantMsg('Address: 789 Pine Rd, Peoria, IL 61602'),
      userMsg('That all looks right'),
      assistantMsg('Great, moving to the next section.'),
      assistantMsg('Entering employment details...'),
    ],
    expectedPreserved: [
      'Rosa Hernandez',
      '1978-11-03',
      'household size',
      '4',             // corrected value
      '1500',          // corrected income
      'TANF',
      'Peoria',
      'caseworker',    // should note it was a caseworker correction
    ],
  },

  {
    name: 'multi-step-with-review',
    description: 'Preserves completed vs pending fields and review state',
    messages: [
      userMsg('File WIC application for participant Sarah Lee'),
      assistantMsg('Looking up participant data...'),
      toolResultMsg('apricot', {
        firstName: 'Sarah',
        lastName: 'Lee',
        dob: '1995-01-20',
        ssn: '***-**-9012',
        address: '321 Elm St, Rockford, IL 61101',
        householdSize: 2,
        monthlyIncome: 2100,
        pregnancyStatus: 'pregnant',
        expectedDueDate: '2026-06-15',
      }),
      assistantMsg(
        'Found Sarah Lee. DOB 1995-01-20, SSN ending 9012. ' +
        'Pregnant, due 2026-06-15. Household: 2, income $2,100.'
      ),
      assistantMsg('Starting WIC application.'),
      toolResultMsg('browser', {
        snapshot: '<html>WIC form page 1</html>',
        url: 'https://benefits.state.il.us/wic/apply/step1',
      }),
      assistantMsg('Page 1: Personal info. Filled name, DOB, SSN.'),
      assistantMsg('Page 2: Pregnancy info. Filled pregnancy status and due date.'),
      toolResultMsg('browser', {
        snapshot: '<html>WIC form page 3</html>',
        url: 'https://benefits.state.il.us/wic/apply/step3',
      }),
      assistantMsg('Page 3: Household/income. Filled household size 2, monthly income $2,100.'),
      assistantMsg(
        'Running form summary:\n' +
        'Completed: name, DOB, SSN, pregnancy status, due date, household size, income\n' +
        'Pending: nutrition assessment, doctor information, preferred WIC office'
      ),
      userMsg('Her doctor is Dr. Amy Chen at Rockford Medical'),
      assistantMsg('Noted — doctor: Dr. Amy Chen, Rockford Medical. Filling that in.'),
      userMsg('Continue to the next section'),
      assistantMsg('Moving to nutrition assessment section...'),
    ],
    expectedPreserved: [
      'Sarah Lee',
      '1995-01-20',
      '9012',
      'pregnant',
      '2026-06-15',
      '2100',
      'WIC',
      'Dr. Amy Chen',
      'Rockford Medical',
      'nutrition assessment', // pending field
      'preferred WIC office', // pending field
    ],
  },
];
