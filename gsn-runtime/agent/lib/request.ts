import { defineState } from 'eve/context';
import type { Preparation } from '../../../lib/gsn/contract';
export const requestState = defineState<Preparation | null>(
  'gsn.preparation',
  () => null,
);
