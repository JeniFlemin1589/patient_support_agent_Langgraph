// lib/langgraph/__tests__/graph.test.ts
import { runHealthcareGraph } from "../graph";
// import type { HealthcareGraphStateType } from "../state";

/**
 * Test the healthcare graph with different scenarios
 * Run with: npx ts-node lib/langgraph/__tests__/graph.test.ts
 */

async function testEmergencyScenario() {
  console.log("\n=== Testing Emergency Scenario ===");

  const input = {
    patientId: "test123",
    query: "I'm having severe chest pain and difficulty breathing",
    chat_history: [],
    user_email: "test@example.com",
  };

  try {
    const result = await runHealthcareGraph(input);

    console.log("Agent Type:", result.agent_type);
    console.log("Severity:", result.severity);
    console.log("Answer:", result.answer?.substring(0, 100) + "...");
    console.log("Email Sent:", result.emailSent);

    // Assertions
    if (result.agent_type !== "emergency") {
      console.error("❌ FAIL: Expected emergency agent");
    } else {
      console.log("✅ PASS: Emergency agent correctly triggered");
    }
  } catch (error) {
    console.error("❌ Test failed:", error);
  }
}

async function testClinicalScenario() {
  console.log("\n=== Testing Clinical Scenario ===");

  const input = {
    patientId: "test456",
    query: "I've had a headache for 3 days. What should I do?",
    chat_history: [],
    user_email: "patient@example.com",
  };

  try {
    const result = await runHealthcareGraph(input);

    console.log("Agent Type:", result.agent_type);
    console.log("Severity:", result.severity);
    console.log("Answer:", result.answer?.substring(0, 100) + "...");
    console.log("Follow-up Questions:", result.followUpQuestions);

    if (result.agent_type !== "clinical") {
      console.error("❌ FAIL: Expected clinical agent");
    } else {
      console.log("✅ PASS: Clinical agent correctly triggered");
    }
  } catch (error) {
    console.error("❌ Test failed:", error);
  }
}

async function testPersonalScenario() {
  console.log("\n=== Testing Personal Scenario ===");

  const input = {
    patientId: "test789",
    query: "Can you show me my previous conversation history?",
    chat_history: [],
    user_email: "user@example.com",
  };

  try {
    const result = await runHealthcareGraph(input);

    console.log("Agent Type:", result.agent_type);
    console.log("Answer:", result.answer?.substring(0, 100) + "...");
    console.log("Needs Email:", result.needsEmail);

    if (result.agent_type !== "personal") {
      console.error("❌ FAIL: Expected personal agent");
    } else {
      console.log("✅ PASS: Personal agent correctly triggered");
    }
  } catch (error) {
    console.error("❌ Test failed:", error);
  }
}

async function testFAQScenario() {
  console.log("\n=== Testing FAQ Scenario ===");

  const input = {
    patientId: "test000",
    query: "What is diabetes and how is it managed?",
    chat_history: [],
  };

  try {
    const result = await runHealthcareGraph(input);

    console.log("Agent Type:", result.agent_type);
    console.log("Answer:", result.answer?.substring(0, 100) + "...");

    if (result.agent_type !== "generic_faq") {
      console.error("❌ FAIL: Expected FAQ agent");
    } else {
      console.log("✅ PASS: FAQ agent correctly triggered");
    }
  } catch (error) {
    console.error("❌ Test failed:", error);
  }
}

// Run all tests
async function runAllTests() {
  console.log("🧪 Starting LangGraph Healthcare Tests\n");

  await testEmergencyScenario();
  await testClinicalScenario();
  await testPersonalScenario();
  await testFAQScenario();

  console.log("\n✅ All tests completed");
}

// Execute tests if run directly
if (require.main === module) {
  runAllTests().catch(console.error);
}

export { runAllTests };
