// lib/agents/index.ts
export {
  supervisorAgent,
  shouldAskFollowUp,
  extractSeverity,
} from "./supervisor";
export { clinicalAgent, emergencyProtocol } from "./clinical-agent";
export { personalAgent } from "./personal-agent";
export { faqAgent } from "./faq-agent";
export { locationAgent } from "./location-agent";
