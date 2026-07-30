import { registerOTel } from '@vercel/otel';
import { BraintrustExporter } from '@braintrust/otel';

export function register() {
  if (!process.env.BRAINTRUST_API_KEY) return;

  registerOTel({
    serviceName: 'labs-asp-chat',
    traceExporter: new BraintrustExporter({ filterAISpans: true }),
  });
}
