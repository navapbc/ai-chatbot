// Participant records the agent fills forms with. There is no participant
// database behind the app: whatever is here is passed to the agent inline in
// the caseworker's message, and the agent treats it as the complete set of
// data it has.

export interface ParticipantAddress {
  street: string;
  unit: string | null;
  city: string;
  state: string;
  county: string;
  zip: string;
  country?: string;
}

export interface Participant {
  record_id: string;
  participant: {
    name: { first: string; middle: string | null; last: string };
    date_of_birth: string;
    participant_type: string;
    funding_source: string;
    calworks_id: string;
    ethnicity: string;
    gender: string;
    primary_language: string;
    special_needs: boolean;
    marital_status: string;
    farm_worker?: boolean;
  };
  dates: {
    file_open_date: string;
    dpss_referral_date: string;
    verbal_consent_provided?: boolean;
  };
  contact_information: {
    preferred_method: string | null;
    phones: {
      home: string | null;
      work: string | null;
      cell: string | null;
      main: string | null;
    };
    email: string;
  };
  address: {
    residential: ParticipantAddress;
    mailing: ParticipantAddress;
  };
  family_profile?: { linked: boolean; notes: string };
}

export const PARTICIPANTS: Participant[] = [
  {
    record_id: '339619',
    participant: {
      name: { first: 'Celeste', middle: 'NAVA', last: 'Thomas II' },
      date_of_birth: '2000-01-02',
      participant_type: 'Parent',
      funding_source: 'FED',
      calworks_id: 'A100103',
      ethnicity: 'Hispanic/Latino',
      gender: 'Female',
      primary_language: 'English',
      special_needs: false,
      marital_status: 'Single parent household',
      farm_worker: false,
    },
    dates: {
      file_open_date: '2025-10-23',
      dpss_referral_date: '2025-10-22',
      verbal_consent_provided: true,
    },
    contact_information: {
      preferred_method: null,
      phones: {
        home: '555-555-5555',
        work: '666-666-6666',
        cell: '777-777-7777',
        main: '888-888-8888',
      },
      email: 'testnava@email.com',
    },
    address: {
      residential: {
        street: '5556 Test Blvd',
        unit: 'Apt 556',
        city: 'WILDOMAR',
        state: 'California',
        county: 'Riverside',
        zip: '92595',
      },
      mailing: {
        street: '5556 Test Blvd',
        unit: 'Apt 556',
        city: 'WILDOMAR',
        state: 'California',
        county: 'Riverside',
        zip: '92595',
      },
    },
    family_profile: {
      linked: false,
      notes: 'No existing Family Profile linked at time of record',
    },
  },
  {
    record_id: '338618',
    participant: {
      name: { first: 'Amelie', middle: 'NAVA', last: 'Thomas I' },
      date_of_birth: '2000-01-01',
      participant_type: 'Parent',
      funding_source: 'Fed',
      calworks_id: 'A100102',
      ethnicity: 'Hispanic/Latino',
      gender: 'Female',
      primary_language: 'English',
      special_needs: false,
      marital_status: 'Single parent household',
    },
    dates: {
      file_open_date: '2025-10-23',
      dpss_referral_date: '2025-10-22',
    },
    contact_information: {
      preferred_method: null,
      phones: {
        home: '5555555555',
        work: '6666666666',
        cell: '7777777777',
        main: '8888888888',
      },
      email: 'testnava@email.com',
    },
    address: {
      residential: {
        street: '5555 Test Blvd',
        unit: 'Apt 555',
        city: 'BANNING',
        state: 'CA',
        county: 'Riverside',
        zip: '92220',
        country: 'US',
      },
      mailing: {
        street: '5555 Test Blvd',
        unit: 'Apt 555',
        city: 'BANNING',
        state: 'CA',
        county: 'Riverside',
        zip: '92220',
        country: 'US',
      },
    },
  },
  {
    record_id: '339637',
    participant: {
      name: { first: 'Sawyer', middle: 'NAVA', last: 'Thomas XX' },
      date_of_birth: '1954-01-10',
      participant_type: 'Other',
      funding_source: 'Fed',
      calworks_id: 'A100121',
      ethnicity: 'Hispanic/Latino',
      gender: 'Male',
      primary_language: 'Spanish',
      special_needs: false,
      marital_status: 'Other',
    },
    dates: {
      file_open_date: '2025-10-23',
      dpss_referral_date: '2025-10-22',
    },
    contact_information: {
      preferred_method: null,
      phones: {
        home: '5555555555',
        work: null,
        cell: '7777777777',
        main: '8888888888',
      },
      email: 'testnava@email.com',
    },
    address: {
      residential: {
        street: '5574 Test Blvd',
        unit: 'Apt 574',
        city: 'WILDOMAR',
        state: 'CA',
        county: 'Riverside',
        zip: '92505',
        country: 'US',
      },
      mailing: {
        street: '5574 Test Blvd',
        unit: 'Apt 574',
        city: 'WILDOMAR',
        state: 'CA',
        county: 'Riverside',
        zip: '92505',
        country: 'US',
      },
    },
  },
];

export const getParticipantById = (recordId: string): Participant | undefined =>
  PARTICIPANTS.find((p) => p.record_id === recordId.trim());

export const getParticipantName = (participant: Participant): string =>
  [
    participant.participant.name.first,
    participant.participant.name.middle,
    participant.participant.name.last,
  ]
    .filter(Boolean)
    .join(' ');

interface ApplicationPromptArgs {
  participant: Participant;
  // Program name, application URL, or both — whatever the caseworker picked.
  target: string;
}

// The single place the participant JSON gets turned into a chat message. Both
// the suggested-action buttons and the landing page go through here so the
// agent always receives the data the same way.
export const buildApplicationPrompt = ({
  participant,
  target,
}: ApplicationPromptArgs): string =>
  `Fill out ${target} for the participant below. This JSON is all the participant data available — there is no database to look up.\n\n${JSON.stringify(
    participant,
    null,
    2,
  )}`;
