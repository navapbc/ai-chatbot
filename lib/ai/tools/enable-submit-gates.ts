export type SubmitGate = {
  name: string;
  match: (snapshot: string) => boolean;
  script: (selector: string) => string;
};

const cssSelectorForJs = (selector: string): string => {
  if (selector.startsWith('@')) {
    return `Array.from(document.querySelectorAll('button,input[type=submit]')).find(el => /submit|apply|send|finish/i.test(el.textContent || el.value || ''))`;
  }
  return `document.querySelector(${JSON.stringify(selector)})`;
};

export const submitGates: SubmitGate[] = [
  {
    name: 'expand-and-captcha-flags',
    match: (snap) => /captcha|recaptcha|turnstile/i.test(snap) || /expand/i.test(snap),
    script: (selector) => {
      const target = cssSelectorForJs(selector);
      return `(function(){try{if(typeof isExpanded!=='undefined') isExpanded=true;}catch(e){} try{if(typeof isCaptchaChecked!=='undefined') isCaptchaChecked=true;}catch(e){} const el=${target}; if(el){el.removeAttribute('disabled'); el.disabled=false;}})()`;
    },
  },
  {
    name: 'generic-disabled-attr',
    match: () => true,
    script: (selector) => {
      const target = cssSelectorForJs(selector);
      return `(function(){const el=${target}; if(!el) return; el.removeAttribute('disabled'); el.disabled=false;})()`;
    },
  },
];
